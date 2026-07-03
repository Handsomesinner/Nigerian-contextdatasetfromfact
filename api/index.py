"""
Vercel serverless demo for Nigerian misinformation detection.

Pure Python standard library only — no numpy / scikit-learn / torch at request
time. It loads the exported TF-IDF + Logistic Regression model (`model.json`,
~19 KB, colocated) and classifies text as misinformation vs. credible, exactly
reproducing scikit-learn's output (verified in src/export_web_model.py).

GET  /              -> HTML demo page
POST /api/predict   -> {"text": "..."} -> {"label", "probability", ...}
"""

import json
import math
import os
import re
from http.server import BaseHTTPRequestHandler

TOKEN_RE = re.compile(r"\b\w\w+\b")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HANDLE_RE = re.compile(r"@\w+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s'#]")
MULTISPACE_RE = re.compile(r"\s+")

_MODEL = None


def load_model():
    global _MODEL
    if _MODEL is None:
        path = os.path.join(os.path.dirname(__file__), "model.json")
        with open(path, encoding="utf-8") as fh:
            _MODEL = json.load(fh)
    return _MODEL


def clean(text):
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = HANDLE_RE.sub(" ", text)
    text = NON_ALNUM_RE.sub(" ", text)
    return MULTISPACE_RE.sub(" ", text).strip()


def ngrams(text, ngram_max):
    tokens = TOKEN_RE.findall(clean(text))
    grams = list(tokens)
    for n in range(2, ngram_max + 1):
        grams += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return grams


def predict(text):
    model = load_model()
    vocab = model["vocab"]           # ngram -> [idf, coef]
    ngram_max = model.get("ngram_max", 2)

    counts = {}
    for g in ngrams(text, ngram_max):
        if g in vocab:
            counts[g] = counts.get(g, 0) + 1

    vec = {g: (1.0 + math.log(c)) * vocab[g][0] for g, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0

    contributions = []
    score = model["intercept"]
    for g, v in vec.items():
        contrib = (v / norm) * vocab[g][1]
        score += contrib
        contributions.append((g, contrib))

    prob = 1.0 / (1.0 + math.exp(-score))
    label = 1 if prob >= 0.5 else 0
    contributions.sort(key=lambda kv: abs(kv[1]), reverse=True)
    return {
        "label": model["labels"][str(label)],
        "label_id": label,
        "probability_misinformation": round(prob, 4),
        "confidence": round(prob if label == 1 else 1 - prob, 4),
        "top_signals": [
            {"ngram": g, "weight": round(c, 4),
             "towards": "misinformation" if c > 0 else "credible"}
            for g, c in contributions[:6]
        ],
        "matched_terms": len(vec),
    }


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nigerian Misinformation Detector</title>
<style>
:root{color-scheme:light dark;--bg:#0f1420;--card:#1b2333;--fg:#e8edf6;--mut:#93a1b8;
--red:#ff5c6c;--green:#31c56d;--acc:#5b8cff;--bd:#2a3549}
@media(prefers-color-scheme:light){:root{--bg:#f4f6fb;--card:#fff;--fg:#141a26;
--mut:#5a667c;--bd:#e2e7f0}}
*{box-sizing:border-box}body{margin:0;font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
background:var(--bg);color:var(--fg);padding:2rem 1rem}
.wrap{max-width:760px;margin:0 auto}
h1{font-size:1.7rem;margin:0 0 .3rem}.sub{color:var(--mut);margin:0 0 1.6rem}
.card{background:var(--card);border:1px solid var(--bd);border-radius:14px;padding:1.3rem;margin-bottom:1.2rem}
textarea{width:100%;min-height:120px;background:transparent;color:var(--fg);border:1px solid var(--bd);
border-radius:10px;padding:.8rem;font:inherit;resize:vertical}
button{margin-top:.9rem;background:var(--acc);color:#fff;border:0;border-radius:10px;
padding:.7rem 1.4rem;font:inherit;font-weight:600;cursor:pointer}
button:disabled{opacity:.6;cursor:wait}
.chips{margin:.6rem 0 0;display:flex;flex-wrap:wrap;gap:.4rem}
.chip{font-size:.82rem;color:var(--mut);border:1px solid var(--bd);border-radius:20px;
padding:.25rem .7rem;cursor:pointer;background:transparent}
.result{display:none}
.verdict{font-size:1.35rem;font-weight:700;display:flex;align-items:center;gap:.5rem}
.bar{height:12px;border-radius:6px;background:var(--bd);overflow:hidden;margin:.7rem 0}
.bar>span{display:block;height:100%}
.meta{color:var(--mut);font-size:.9rem}
.sig{display:flex;justify-content:space-between;padding:.35rem 0;border-top:1px solid var(--bd);font-size:.92rem}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace}
.foot{color:var(--mut);font-size:.82rem;margin-top:1.4rem;text-align:center}
</style></head><body><div class="wrap">
<h1>Nigerian Misinformation Detector</h1>
<p class="sub">TF-IDF + Logistic Regression, trained on a Nigerian-context fact-check corpus.
Educational demo — not a verdict on any real claim.</p>
<div class="card">
<textarea id="t" placeholder="Paste a claim or social-media post, e.g. 'Drinking warm salt water flushes out coronavirus'..."></textarea>
<div class="chips" id="ex"></div>
<button id="b" onclick="go()">Analyse</button>
</div>
<div class="card result" id="r">
<div class="verdict" id="v"></div>
<div class="bar"><span id="bar"></span></div>
<div class="meta" id="m"></div>
<div id="sigs" style="margin-top:.9rem"></div>
</div>
<p class="foot">Model reproduces scikit-learn output exactly (pure-Python inference).
Part of the "ML Approach to Detecting Misinformation in Nigerian Social Media Content" project.</p>
</div>
<script>
const EX=["The COVID-19 vaccine contains a 5G microchip to track Nigerians who take it.",
"INEC introduced the BVAS to verify voters using fingerprints and facial recognition.",
"Forward this message to 10 people and the government will send you N30,000 palliative.",
"Malaria is transmitted through the bite of an infected female Anopheles mosquito."];
const ex=document.getElementById('ex');
EX.forEach(e=>{const c=document.createElement('span');c.className='chip';
c.textContent=e.length>52?e.slice(0,52)+'…':e;c.onclick=()=>{document.getElementById('t').value=e;go()};ex.appendChild(c)});
async function go(){
 const t=document.getElementById('t').value.trim();if(!t)return;
 const b=document.getElementById('b');b.disabled=true;b.textContent='Analysing…';
 try{
  const res=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
  const d=await res.json();
  const mis=d.label_id===1;const pct=Math.round(d.probability_misinformation*100);
  document.getElementById('r').style.display='block';
  document.getElementById('v').innerHTML=(mis?'🚩 ':'✅ ')+
   '<span style="color:'+(mis?'var(--red)':'var(--green)')+'">'+d.label.toUpperCase()+'</span>';
  document.getElementById('bar').style.width=pct+'%';
  document.getElementById('bar').style.background=mis?'var(--red)':'var(--green)';
  document.getElementById('m').textContent='P(misinformation) = '+pct+'%  ·  confidence '+
   Math.round(d.confidence*100)+'%  ·  '+d.matched_terms+' known terms matched';
  let h=d.top_signals.length?'<div class="meta">Top signals</div>':'';
  d.top_signals.forEach(s=>{h+='<div class="sig"><span class="mono">'+s.ngram+
   '</span><span style="color:'+(s.weight>0?'var(--red)':'var(--green)')+'">'+
   (s.weight>0?'+':'')+s.weight+' → '+s.towards+'</span></div>'});
  document.getElementById('sigs').innerHTML=h;
 }catch(e){document.getElementById('r').style.display='block';
  document.getElementById('v').textContent='Error: '+e}
 b.disabled=false;b.textContent='Analyse';
}
</script></body></html>"""


class handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") in ("/api/predict",):
            self._send(405, json.dumps({"error": "use POST"}), "application/json")
        else:
            self._send(200, PAGE, "text/html; charset=utf-8")

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = (payload.get("text") or "").strip()
            if not text:
                self._send(400, json.dumps({"error": "field 'text' required"}),
                           "application/json")
                return
            result = predict(text)
            self._send(200, json.dumps(result), "application/json")
        except Exception as exc:  # pragma: no cover
            self._send(500, json.dumps({"error": str(exc)}), "application/json")
