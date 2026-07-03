"""
Nigerian misinformation *verification* service (Vercel serverless, stdlib-only).

Given free text, a news URL, or a social-media link, it:

  1. extracts the claim (fetches + parses the page for links),
  2. retrieves live evidence from established fact-checkers via the Google
     Fact Check Tools API (ClaimReview: Africa Check, Dubawa, PolitiFact, ...),
  3. asks the Anthropic API (Claude) for a reasoned assessment,
  4. always runs the local TF-IDF model as an offline fallback signal,

then combines them into one verdict with clickable sources.

No third-party packages: URL fetching and both APIs go through urllib. Secrets
come from environment variables set in the Vercel dashboard:

    GOOGLE_FACTCHECK_API_KEY   free key from Google Cloud (Fact Check Tools API)
    ANTHROPIC_API_KEY          key from console.anthropic.com
    ANTHROPIC_MODEL            optional, default "claude-haiku-4-5-20251001"

Everything degrades gracefully: with no keys it behaves as the offline ML demo.

Routes:  GET /   -> HTML UI
         POST /api/verify   {"input": "..."}  -> full hybrid verdict
         POST /api/predict  {"text": "..."}   -> ML-only signal (legacy)
"""

import datetime as _dt
import ipaddress
import json
import math
import os
import re
import socket
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from http.server import BaseHTTPRequestHandler

# --------------------------------------------------------------------------- #
#  Local TF-IDF model (offline fallback)                                       #
# --------------------------------------------------------------------------- #
TOKEN_RE = re.compile(r"\b\w\w+\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HANDLE_RE = re.compile(r"@\w+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s'#]")
MULTISPACE_RE = re.compile(r"\s+")
IS_URL_RE = re.compile(r"^\s*https?://", re.I)

MIN_MATCHED_TERMS = 3
UNCERTAIN_BAND = 0.10
_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is None:
        with open(os.path.join(os.path.dirname(__file__), "model.json"),
                  encoding="utf-8") as fh:
            _MODEL = json.load(fh)
    return _MODEL


def clean(text):
    text = URL_RE.sub(" ", str(text).lower())
    text = HANDLE_RE.sub(" ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    return MULTISPACE_RE.sub(" ", text).strip()


def ngrams(text, ngram_max):
    tokens = TOKEN_RE.findall(clean(text))
    grams = list(tokens)
    for n in range(2, ngram_max + 1):
        grams += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return grams


def ml_predict(text):
    model = load_model()
    vocab = model["vocab"]
    counts = {}
    for g in ngrams(text, model.get("ngram_max", 2)):
        if g in vocab:
            counts[g] = counts.get(g, 0) + 1
    vec = {g: (1.0 + math.log(c)) * vocab[g][0] for g, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    contribs = []
    score = model["intercept"]
    for g, v in vec.items():
        c = (v / norm) * vocab[g][1]
        score += c
        contribs.append((g, c))
    prob = 1.0 / (1.0 + math.exp(-score))
    raw = 1 if prob >= 0.5 else 0
    contribs.sort(key=lambda kv: abs(kv[1]), reverse=True)
    matched = len(vec)
    uncertain = matched < MIN_MATCHED_TERMS or abs(prob - 0.5) < UNCERTAIN_BAND
    return {
        "verdict": "uncertain" if uncertain else model["labels"][str(raw)],
        "uncertain": uncertain,
        "probability_misinformation": round(prob, 4),
        "matched_terms": matched,
        "top_signals": [
            {"ngram": g, "weight": round(c, 4),
             "towards": "misinformation" if c > 0 else "credible"}
            for g, c in contribs[:6]],
    }


# --------------------------------------------------------------------------- #
#  URL fetching + claim extraction                                             #
# --------------------------------------------------------------------------- #
class _Extract(HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head"}
    KEEP = {"p", "h1", "h2", "h3", "article", "li", "blockquote"}

    def __init__(self):
        super().__init__()
        self.title = ""
        self.og = {}
        self._skip = 0
        self._keep = 0
        self._in_title = False
        self.chunks = []

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in self.KEEP:
            self._keep += 1
        if tag == "meta":
            a = dict(attrs)
            key = (a.get("property") or a.get("name") or "").lower()
            if key in ("og:title", "og:description", "twitter:title",
                       "twitter:description", "description") and a.get("content"):
                self.og.setdefault(key, a["content"].strip())

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False
        if tag in self.KEEP and self._keep:
            self._keep -= 1

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title += data
        elif self._keep:
            t = data.strip()
            if len(t) > 1:
                self.chunks.append(t)


def _url_is_safe(url):
    try:
        p = urllib.parse.urlparse(url)
        if p.scheme not in ("http", "https") or not p.hostname:
            return False
        for fam, _, _, _, sa in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(sa[0])
            if (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_multicast):
                return False
        return True
    except Exception:
        return False


def fetch_claim(url):
    """Fetch a URL and return (claim_text, context_text, meta) or raise."""
    if not _url_is_safe(url):
        raise ValueError("URL is not reachable or not allowed.")
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; NG-MisinfoVerify/1.0)",
        "Accept": "text/html,application/xhtml+xml"})
    with urllib.request.urlopen(req, timeout=8) as resp:
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read(1_500_000)
    if "html" not in ctype and "xml" not in ctype and not raw.lstrip().startswith(b"<"):
        text = raw.decode("utf-8", "ignore")
        return text[:400], text[:2000], {"final_url": url}
    charset = "utf-8"
    m = re.search(r"charset=([\w-]+)", ctype)
    if m:
        charset = m.group(1)
    ex = _Extract()
    try:
        ex.feed(raw.decode(charset, "ignore"))
    except Exception:
        ex.feed(raw.decode("utf-8", "ignore"))
    claim = (ex.og.get("og:title") or ex.og.get("twitter:title")
             or ex.title.strip() or "")
    desc = (ex.og.get("og:description") or ex.og.get("twitter:description")
            or ex.og.get("description") or "")
    body = " ".join(ex.chunks)[:2000]
    context = (desc + " " + body).strip()[:2000]
    if not claim:
        claim = context[:200]
    return claim.strip()[:400], context, {"final_url": url, "title": ex.title.strip()}


# --------------------------------------------------------------------------- #
#  Evidence source 1: Google Fact Check Tools API                              #
# --------------------------------------------------------------------------- #
RATING_MAP = [
    ("pants on fire", "false"), ("false", "false"), ("incorrect", "false"),
    ("fake", "false"), ("hoax", "false"), ("no evidence", "false"),
    ("misleading", "misleading"), ("mixture", "misleading"),
    ("partly", "misleading"), ("half", "misleading"), ("exaggerat", "misleading"),
    ("unproven", "misleading"), ("misattributed", "misleading"),
    ("mostly true", "true"), ("true", "true"), ("correct", "true"),
    ("accurate", "true"), ("legit", "true"),
]


def normalise_rating(text):
    t = (text or "").lower()
    for key, val in RATING_MAP:
        if key in t:
            return val
    return None


def google_factcheck(query):
    key = os.environ.get("GOOGLE_FACTCHECK_API_KEY")
    if not key or not query.strip():
        return []
    params = urllib.parse.urlencode({
        "query": query[:300], "key": key, "languageCode": "en", "pageSize": 6})
    url = "https://factchecktools.googleapis.com/v1alpha1/claims:search?" + params
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
    except Exception:
        return []
    out = []
    for claim in data.get("claims", []):
        for rev in claim.get("claimReview", []):
            rating = rev.get("textualRating")
            out.append({
                "claim": claim.get("text"),
                "rating": rating,
                "normalised": normalise_rating(rating),
                "publisher": (rev.get("publisher") or {}).get("name"),
                "url": rev.get("url"),
                "title": rev.get("title"),
            })
    return out


# --------------------------------------------------------------------------- #
#  Evidence source 2: Anthropic (Claude) reasoning                             #
# --------------------------------------------------------------------------- #
LLM_PROMPT = """You are a careful, neutral fact-checking assistant with a focus \
on Nigerian and African context. Today's date is {today}. Assess the CLAIM below.

You have a web_search tool. USE IT whenever the claim could depend on recent \
events, current status, deaths, elections, appointments, prices, or any fact that \
may have changed after your training cutoff — do not rely on memory for anything \
time-sensitive. Search for the latest reporting from credible outlets, then judge. \
Be sceptical of sensational health cures, giveaways, and doctored quotes. If, even \
after searching, you cannot confirm the claim, answer "unverifiable" rather than \
guessing.

After any searches, end your reply with ONLY a JSON object on its own, in this \
exact shape (no extra prose after it):
{{"verdict": "true|false|misleading|unverifiable", "confidence": 0.0-1.0, \
"explanation": "2-3 plain sentences a general reader understands", \
"reasoning_points": ["short point", "short point"]}}

CLAIM: {claim}
{context}"""


def _anthropic_call(key, model, prompt, use_tools):
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    if use_tools:
        payload["tools"] = [{"type": "web_search_20250305", "name": "web_search",
                             "max_uses": 5}]
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(payload).encode(),
        method="POST",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=55) as resp:
        return json.loads(resp.read())


def _parse_anthropic(data, model, web_search):
    content = data.get("content", [])
    text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
    sources, seen = [], set()
    for b in content:
        if b.get("type") == "web_search_tool_result":
            for r in b.get("content", []):
                url = r.get("url")
                if r.get("type") == "web_search_result" and url and url not in seen:
                    seen.add(url)
                    sources.append({"title": r.get("title") or url, "url": url})
    m = re.search(r"\{.*\}", text, re.S)
    parsed = json.loads(m.group(0)) if m else {}
    return {
        "verdict": parsed.get("verdict", "unverifiable"),
        "confidence": parsed.get("confidence"),
        "explanation": parsed.get("explanation", "").strip(),
        "reasoning_points": parsed.get("reasoning_points", [])[:5],
        "web_sources": sources[:6],
        "web_search": web_search and bool(sources),
        "model": model,
    }


def anthropic_assess(claim, context=""):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key or not claim.strip():
        return None
    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    ctx = f"\nARTICLE CONTEXT: {context[:1200]}" if context else ""
    today = _dt.datetime.utcnow().strftime("%d %B %Y")
    prompt = LLM_PROMPT.format(today=today, claim=claim[:1500], context=ctx)
    want_search = os.environ.get("ANTHROPIC_WEB_SEARCH", "1") != "0"
    try:
        try:
            data = _anthropic_call(key, model, prompt, use_tools=want_search)
            return _parse_anthropic(data, model, web_search=want_search)
        except urllib.error.HTTPError as e:
            # e.g. web search not enabled on the account -> retry without tools
            if want_search and e.code in (400, 403, 404):
                data = _anthropic_call(key, model, prompt, use_tools=False)
                out = _parse_anthropic(data, model, web_search=False)
                out["note"] = "Web search unavailable on this key; answered from model knowledge only."
                return out
            raise
    except urllib.error.HTTPError as e:
        return {"error": f"Anthropic API error {e.code}",
                "detail": e.read().decode("utf-8", "ignore")[:300]}
    except Exception as e:
        return {"error": str(e)}


# --------------------------------------------------------------------------- #
#  Combine into a single verdict                                               #
# --------------------------------------------------------------------------- #
DISPLAY = {"false": ("misinformation", "🚩"), "misleading": ("misleading", "⚠️"),
           "true": ("credible", "✅"), "unverifiable": ("unverifiable", "🤔"),
           "uncertain": ("uncertain", "🤔"), "misinformation": ("misinformation", "🚩"),
           "credible": ("credible", "✅")}


def _factcheck_consensus(results):
    votes = [r["normalised"] for r in results if r["normalised"]]
    if not votes:
        return None
    for v in ("false", "misleading", "true"):   # prioritise a "false" signal
        if votes.count(v) == max(votes.count(x) for x in set(votes)) and v in votes:
            return v
    return votes[0]


def verify(user_input):
    user_input = (user_input or "").strip()
    is_url = bool(IS_URL_RE.match(user_input))
    source_url = user_input if is_url else None
    context = ""
    fetch_error = None

    if is_url:
        try:
            claim, context, _ = fetch_claim(user_input)
        except Exception as e:
            claim, fetch_error = user_input, str(e)
    else:
        claim = user_input

    factchecks = google_factcheck(claim)
    llm = anthropic_assess(claim, context)
    ml = ml_predict(claim)

    # Decide the headline verdict, best evidence first.
    sources_used = []
    fc_consensus = _factcheck_consensus(factchecks)
    if fc_consensus:
        verdict, basis = fc_consensus, "published fact-checks"
        confidence = 0.9
        sources_used.append("Google Fact Check")
    elif llm and not llm.get("error") and llm.get("verdict"):
        searched = llm.get("web_search")
        verdict = llm["verdict"]
        basis = "Claude + live web search" if searched else "AI reasoning (Claude)"
        confidence = llm.get("confidence") or (0.7 if searched else 0.6)
        sources_used.append("Claude web search" if searched else "Claude")
    else:
        verdict, basis = ml["verdict"], "offline ML model (limited)"
        confidence = ml["probability_misinformation"] if ml["verdict"] != "uncertain" else None
        sources_used.append("local ML")

    label, icon = DISPLAY.get(verdict, (verdict, "🤔"))
    keys_present = {
        "google_factcheck": bool(os.environ.get("GOOGLE_FACTCHECK_API_KEY")),
        "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }
    return {
        "input_type": "url" if is_url else "text",
        "source_url": source_url,
        "claim": claim,
        "fetch_error": fetch_error,
        "verdict": verdict,
        "display_label": label,
        "icon": icon,
        "basis": basis,
        "confidence": confidence,
        "sources_used": sources_used,
        "fact_checks": factchecks,
        "llm": llm,
        "ml": ml,
        "keys_present": keys_present,
    }


# --------------------------------------------------------------------------- #
#  HTML UI                                                                      #
# --------------------------------------------------------------------------- #
PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nigerian Claim & News Verifier</title>
<style>
:root{color-scheme:light dark;--bg:#0f1420;--card:#1b2333;--fg:#e8edf6;--mut:#93a1b8;
--red:#ff5c6c;--green:#31c56d;--amber:#e0a338;--acc:#5b8cff;--bd:#2a3549}
@media(prefers-color-scheme:light){:root{--bg:#f4f6fb;--card:#fff;--fg:#141a26;
--mut:#5a667c;--bd:#e2e7f0}}
*{box-sizing:border-box}body{margin:0;font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
background:var(--bg);color:var(--fg);padding:2rem 1rem}
.wrap{max-width:780px;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 .3rem}.sub{color:var(--mut);margin:0 0 1.4rem}
.card{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:1.3rem;margin-bottom:1.1rem}
textarea{width:100%;min-height:96px;background:transparent;color:var(--fg);border:1px solid var(--bd);
border-radius:10px;padding:.8rem;font:inherit;resize:vertical}
button{margin-top:.8rem;background:var(--acc);color:#fff;border:0;border-radius:10px;
padding:.7rem 1.4rem;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}
.chips{margin:.6rem 0 0;display:flex;flex-wrap:wrap;gap:.4rem}
.chip{font-size:.82rem;color:var(--mut);border:1px solid var(--bd);border-radius:20px;
padding:.25rem .7rem;cursor:pointer;background:transparent}
.result{display:none}
.verdict{font-size:1.4rem;font-weight:700;display:flex;align-items:center;gap:.5rem}
.claim{color:var(--mut);font-size:.92rem;margin:.5rem 0 .2rem}
.basis{font-size:.85rem;color:var(--mut);margin-top:.3rem}
.sec{margin-top:1rem;border-top:1px solid var(--bd);padding-top:.8rem}
.sec h3{font-size:.8rem;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);margin:0 0 .5rem}
.ai{margin-top:1.1rem;background:rgba(91,140,255,.09);border:1px solid var(--acc);
border-radius:12px;padding:1rem 1.1rem}
.ai h3{color:var(--acc);margin:0 0 .5rem}
.ai .exp{font-size:1.1rem;line-height:1.6}
.ai .call{font-size:.85rem;color:var(--mut);margin-top:.5rem}
.muted{opacity:.62}.muted .mono{font-size:.82rem}
.fc{padding:.5rem 0;border-bottom:1px solid var(--bd)}
.fc a{color:var(--acc);text-decoration:none}.fc a:hover{text-decoration:underline}
.pill{display:inline-block;font-size:.72rem;font-weight:700;padding:.1rem .5rem;border-radius:20px;margin-right:.4rem}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:.9rem}
.warn{background:rgba(224,163,56,.12);border:1px solid var(--amber);border-radius:10px;
padding:.6rem .8rem;font-size:.85rem;margin-bottom:1rem}
.foot{color:var(--mut);font-size:.8rem;margin-top:1.3rem;text-align:center}
ul{margin:.3rem 0 0 1.1rem;padding:0}
</style></head><body><div class="wrap">
<h1>Nigerian Claim &amp; News Verifier</h1>
<p class="sub">Paste a claim, a news article link, or a social-media link. It checks
established fact-checkers, searches the live web, reasons over the claim, and shows
its sources. Verdicts are guidance, not proof.</p>
<div id="setup"></div>
<div class="card">
<textarea id="t" placeholder="Paste a statement, or a link like https://... (news or social post)"></textarea>
<div class="chips" id="ex"></div>
<button id="b" onclick="go()">Verify</button>
</div>
<div class="card result" id="r">
<div class="verdict" id="v"></div>
<div class="basis" id="basis"></div>
<div class="claim" id="claim"></div>
<div id="body"></div>
</div>
<p class="foot">Hybrid verifier: Google Fact Check Tools API + Claude with live web search +
an offline TF-IDF model. Educational project — always check the linked sources yourself.</p>
</div>
<script>
const EX=["https://www.bbc.com/news",
"The COVID-19 vaccine contains a 5G microchip to track Nigerians who take it.",
"INEC introduced the BVAS to verify voters using fingerprints and facial recognition.",
"Forward this message to 10 people and the government will send you N30,000 palliative."];
const ex=document.getElementById('ex');
EX.forEach(e=>{const c=document.createElement('span');c.className='chip';
c.textContent=e.length>50?e.slice(0,50)+'…':e;c.onclick=()=>{document.getElementById('t').value=e;go()};ex.appendChild(c)});
const COL={misinformation:'var(--red)',misleading:'var(--amber)',credible:'var(--green)',
uncertain:'var(--amber)',unverifiable:'var(--amber)'};
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function go(){
 const t=document.getElementById('t').value.trim();if(!t)return;
 const b=document.getElementById('b');b.disabled=true;b.textContent='Verifying…';
 try{
  const res=await fetch('/api/verify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({input:t})});
  const d=await res.json();render(d);
 }catch(e){document.getElementById('r').style.display='block';
  document.getElementById('v').textContent='Error: '+e}
 b.disabled=false;b.textContent='Verify';
}
function render(d){
 document.getElementById('r').style.display='block';
 const col=COL[d.display_label]||'var(--mut)';
 document.getElementById('v').innerHTML=d.icon+' <span style="color:'+col+'">'+esc(d.display_label.toUpperCase())+'</span>'+
   (d.confidence!=null?' <span style="font-size:.85rem;color:var(--mut)">('+Math.round(d.confidence*100)+'% conf.)</span>':'');
 document.getElementById('basis').textContent='Based on: '+esc(d.basis)+' · sources: '+(d.sources_used||[]).join(', ');
 document.getElementById('claim').innerHTML=(d.input_type==='url'?'🔗 Extracted claim: ':'')+
   '“'+esc(d.claim)+'”'+(d.fetch_error?' <span style="color:var(--amber)">(could not fetch link: '+esc(d.fetch_error)+')</span>':'');
 let h='';
 // fact-checks
 if(d.fact_checks&&d.fact_checks.length){
  h+='<div class="sec"><h3>Published fact-checks</h3>';
  d.fact_checks.forEach(f=>{const c=COL[({false:'misinformation',misleading:'misleading',true:'credible'})[f.normalised]]||'var(--mut)';
   h+='<div class="fc"><span class="pill" style="background:'+c+';color:#fff">'+esc(f.rating||'?')+'</span>'+
      '<a href="'+esc(f.url)+'" target="_blank" rel="noopener">'+esc(f.publisher||f.title||f.url)+'</a>'+
      (f.claim?'<div class="mono" style="color:var(--mut)">“'+esc(f.claim)+'”</div>':'')+'</div>'});
  h+='</div>';
 }
 // llm — the centrepiece when Claude answered
 const hasLlm=d.llm&&!d.llm.error&&d.llm.explanation;
 if(hasLlm){
  const lc=COL[({true:'credible',false:'misinformation',misleading:'misleading',unverifiable:'unverifiable'})[d.llm.verdict]]||'var(--mut)';
  const badge=d.llm.web_search?' <span class="pill" style="background:var(--acc);color:#fff">🔎 live web</span>':'';
  h+='<div class="ai"><h3>🧠 AI assessment (Claude)'+badge+'</h3><div class="exp">'+esc(d.llm.explanation)+'</div>';
  if(d.llm.reasoning_points&&d.llm.reasoning_points.length){h+='<ul>'+d.llm.reasoning_points.map(p=>'<li>'+esc(p)+'</li>').join('')+'</ul>'}
  if(d.llm.web_sources&&d.llm.web_sources.length){
    h+='<div class="call">Sources Claude checked:</div><div>'+d.llm.web_sources.map(s=>
      '<div class="fc"><a href="'+esc(s.url)+'" target="_blank" rel="noopener">'+esc(s.title)+'</a></div>').join('')+'</div>';
  }
  if(d.llm.note){h+='<div class="call">⚠️ '+esc(d.llm.note)+'</div>';}
  h+='<div class="call">Claude\\'s call: <b style="color:'+lc+'">'+esc((d.llm.verdict||'').toUpperCase())+'</b>'+
     (d.llm.confidence!=null?' · '+Math.round(d.llm.confidence*100)+'% confidence':'')+
     (d.llm.model?' · '+esc(d.llm.model):'')+'</div></div>';
 } else if(d.llm&&d.llm.error){
  h+='<div class="sec"><h3>AI assessment</h3><div style="color:var(--amber)">'+esc(d.llm.error)+
     '</div><div class="basis">Check that ANTHROPIC_API_KEY (and model access) are set in Vercel.</div></div>';
 }
 // ml — primary when nothing else answered, otherwise a muted technical footnote
 if(d.ml){
  const mlPrimary=!hasLlm&&!(d.fact_checks&&d.fact_checks.length);
  h+='<div class="sec'+(mlPrimary?'':' muted')+'"><h3>'+(mlPrimary?'Offline ML signal':'Technical fallback signal (offline model)')+
   '</h3><div class="mono">'+esc(d.ml.verdict)+' · P(misinfo)='+d.ml.probability_misinformation+
   ' · '+d.ml.matched_terms+' known terms</div></div>';
 }
 document.getElementById('body').innerHTML=h;
 // setup hint if no keys
 const s=document.getElementById('setup');
 if(d.keys_present&&!(d.keys_present.google_factcheck||d.keys_present.anthropic)){
  s.innerHTML='<div class="warn">⚙️ Running in <b>offline mode</b>. Add <span class="mono">GOOGLE_FACTCHECK_API_KEY</span> and/or '+
   '<span class="mono">ANTHROPIC_API_KEY</span> in your Vercel project settings to enable live fact-check retrieval and AI reasoning.</div>';
 } else {s.innerHTML='';}
}
</script></body></html>"""


# --------------------------------------------------------------------------- #
#  HTTP handler                                                                 #
# --------------------------------------------------------------------------- #
class handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep Vercel logs quiet
        pass

    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/api/verify", "/api/predict"):
            self._send(405, json.dumps({"error": "use POST"}), "application/json")
        else:
            self._send(200, PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        try:
            payload = self._read_json()
            if path == "/api/predict":
                text = (payload.get("text") or "").strip()
                if not text:
                    return self._send(400, json.dumps({"error": "field 'text' required"}), "application/json")
                return self._send(200, json.dumps(ml_predict(text)), "application/json")
            # default: full verification
            user_input = (payload.get("input") or payload.get("text") or "").strip()
            if not user_input:
                return self._send(400, json.dumps({"error": "field 'input' required"}), "application/json")
            self._send(200, json.dumps(verify(user_input)), "application/json")
        except Exception as exc:  # pragma: no cover
            self._send(500, json.dumps({"error": str(exc)}), "application/json")
