"""
Construction of a Nigerian-context labelled dataset for misinformation detection.

The corpus is curated to mirror the categories, claim styles and verdicts
published by established Nigerian / African fact-checking desks:

    * Africa Check        (https://africacheck.org)
    * Dubawa             (https://dubawa.org)
    * FactCheckHub       (https://factcheckhub.com)  -- ICIR
    * PesaCheck          (https://pesacheck.org)
    * The Cable / TheCable Fact Check

IMPORTANT / PROVENANCE
----------------------
The records below are a *curated, illustrative* research corpus. Each item is
written to reflect a real theme, claim register and published verdict that these
organisations have reported on (COVID-19 cures, election rumours, the 2023 naira
redesign, fuel-subsidy claims, insecurity, doctored quotes attributed to public
figures, health myths, online financial scams, etc.). It is intended to make the
end-to-end modelling pipeline fully reproducible *offline*. It is NOT a scrape of
any organisation's copyrighted article text. `src/scrape.py` provides a scaffold
for collecting live articles where terms of use permit.

Label scheme
------------
    verdict  : the fine-grained rating  -> {false, misleading, true}
    label    : the binary target        -> 1 = misinformation (false|misleading)
                                            0 = credible       (true)
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict
from typing import List

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
OUT_CSV = os.path.join(RAW_DIR, "nigerian_factcheck_dataset.csv")


@dataclass
class Record:
    text: str
    source: str
    category: str
    verdict: str  # false | misleading | true

    @property
    def label(self) -> int:
        return 0 if self.verdict == "true" else 1


# ---------------------------------------------------------------------------
# MISINFORMATION  (verdict = false | misleading)
# ---------------------------------------------------------------------------
MISINFO: List[Record] = [
    # --- COVID-19 / health myths -------------------------------------------
    Record("Drinking warm water mixed with lemon and salt flushes the coronavirus "
           "out of your throat before it reaches the lungs, Nigerian doctors confirm.",
           "Dubawa", "health", "false"),
    Record("BREAKING: The COVID-19 vaccine contains a 5G microchip that the "
           "government is using to track every Nigerian who takes it.",
           "AfricaCheck", "health", "false"),
    Record("Bathing in hot water and steam inhalation kills the coronavirus and "
           "makes vaccination completely unnecessary.",
           "FactCheckHub", "health", "false"),
    Record("A Nigerian herbalist has produced a herbal mixture that cures COVID-19 "
           "in just three days with a 100% success rate.",
           "Dubawa", "health", "false"),
    Record("Africans are genetically immune to the coronavirus because of the hot "
           "weather and high melanin levels.",
           "AfricaCheck", "health", "false"),
    Record("Eating plenty of garlic and drinking chloroquine tea prevents anyone "
           "from ever catching COVID-19.",
           "FactCheckHub", "health", "misleading"),
    Record("The polio vaccine being distributed in northern Nigeria was designed to "
           "make young girls infertile, a viral audio claims.",
           "AfricaCheck", "health", "false"),
    Record("Photos show that the COVID-19 test swab inserts a nanoworm into the "
           "brain through the nose.",
           "Dubawa", "health", "false"),
    Record("Doctors in Lagos say drinking your own urine every morning boosts your "
           "immune system enough to defeat any virus.",
           "FactCheckHub", "health", "false"),
    Record("A message on WhatsApp says NAFDAC has approved a N200 tea that reverses "
           "diabetes and high blood pressure permanently.",
           "Dubawa", "health", "false"),
    Record("Sitting in the sun for 30 minutes at noon raises your body temperature "
           "high enough to kill the coronavirus in your blood.",
           "AfricaCheck", "health", "false"),
    Record("Fresh reports claim that the malaria drug Artemisia tea eliminates HIV "
           "from the body within two weeks.",
           "PesaCheck", "health", "false"),
    Record("Ebola can be prevented by drinking large amounts of salt water, a "
           "circulating broadcast tells Nigerians.",
           "AfricaCheck", "health", "false"),
    Record("Applying a mixture of onion and honey to the chest overnight cures "
           "pneumonia without any need for antibiotics.",
           "FactCheckHub", "health", "misleading"),
    Record("A viral post insists that Monkeypox in Nigeria is actually a side effect "
           "of the COVID-19 vaccine, not a separate virus.",
           "Dubawa", "health", "false"),

    # --- Elections / politics ----------------------------------------------
    Record("INEC has secretly declared the election results 24 hours before voting "
           "even began, leaked documents show.",
           "FactCheckHub", "elections", "false"),
    Record("A viral video shows thumb-printed ballot papers being burned by soldiers "
           "in Rivers State during the 2023 general election.",
           "Dubawa", "elections", "misleading"),
    Record("The BVAS machines used by INEC can be remotely reprogrammed overnight to "
           "flip votes to a preferred candidate.",
           "AfricaCheck", "elections", "false"),
    Record("Photo shows that a foreign country has already printed congratulatory "
           "banners for a candidate before Nigerians voted.",
           "FactCheckHub", "elections", "misleading"),
    Record("A doctored INEC statement announces that the presidential election has "
           "been postponed by three months.",
           "Dubawa", "elections", "false"),
    Record("Millions of underage children were bussed in to vote in Kano, according "
           "to a widely shared but unverified clip.",
           "AfricaCheck", "elections", "misleading"),
    Record("The Chairman of INEC has resigned live on television after admitting the "
           "election was rigged, a fake screenshot claims.",
           "TheCable", "elections", "false"),
    Record("A recycled 2019 video of ballot-box snatching is being shared as fresh "
           "evidence of fraud in the 2023 governorship poll.",
           "Dubawa", "elections", "misleading"),
    Record("Viral message: the Central Bank will pay every registered voter N50,000 "
           "as an incentive for turning out on election day.",
           "FactCheckHub", "elections", "false"),
    Record("Satellite images supposedly prove that entire polling units in Lagos "
           "were fabricated and never existed.",
           "AfricaCheck", "elections", "false"),

    # --- Naira redesign / economy / CBN ------------------------------------
    Record("The CBN has announced that the old N1,000 notes will remain legal tender "
           "forever and should never be returned to banks.",
           "PesaCheck", "economy", "false"),
    Record("A viral flyer claims the government will convert all naira savings to a "
           "digital currency and wipe out physical cash balances next month.",
           "Dubawa", "economy", "false"),
    Record("New N5,000 and N10,000 naira notes with a president's portrait are "
           "already in circulation, according to shared photos.",
           "FactCheckHub", "economy", "false"),
    Record("The IMF has ordered Nigeria to seize 40% of every citizen's bank balance "
           "to repay national debt, a message warns.",
           "AfricaCheck", "economy", "false"),
    Record("A circulating notice says petrol will sell for N100 per litre nationwide "
           "starting next week after a secret deal.",
           "Dubawa", "economy", "misleading"),
    Record("The fuel subsidy removal has been reversed and pump prices will crash "
           "back to N165, an unverified broadcast claims.",
           "FactCheckHub", "economy", "misleading"),
    Record("A fake CBN circular says all dormant bank accounts will be emptied and "
           "the money donated to charity by Friday.",
           "PesaCheck", "economy", "false"),
    Record("Message claims the World Bank has cancelled all of Nigeria's debt as a "
           "gift, so taxes will be abolished this year.",
           "AfricaCheck", "economy", "false"),

    # --- Insecurity / disaster ---------------------------------------------
    Record("A video of a building collapse in another country is being shared as a "
           "fresh bomb explosion in Abuja this morning.",
           "Dubawa", "security", "misleading"),
    Record("Bandits have taken over the entire Kaduna-Abuja highway and no vehicle "
           "has passed in two weeks, a viral voice note claims.",
           "FactCheckHub", "security", "misleading"),
    Record("Old footage from Iraq is circulating as proof that a Nigerian city has "
           "been captured by insurgents overnight.",
           "AfricaCheck", "security", "false"),
    Record("A message warns that armed men are injecting a deadly substance into "
           "sachet water sold at Lagos motor parks.",
           "Dubawa", "security", "false"),
    Record("Viral claim: the army has declared a 72-hour nationwide curfew and "
           "anyone outside will be arrested on sight.",
           "FactCheckHub", "security", "false"),
    Record("A staged photo is being passed around as evidence that a governor was "
           "kidnapped from his convoy last night.",
           "AfricaCheck", "security", "misleading"),

    # --- Doctored quotes / impersonation -----------------------------------
    Record("A fabricated quote card claims the President said Nigerians should 'eat "
           "grass' if they cannot afford rice.",
           "Dubawa", "politics", "false"),
    Record("A fake tweet attributed to Elon Musk promises to give N1 billion to any "
           "Nigerian who retweets and sends their bank details.",
           "FactCheckHub", "scam", "false"),
    Record("A doctored BBC News graphic claims Nigeria has been ranked the happiest "
           "country in the world for 2023.",
           "AfricaCheck", "politics", "false"),
    Record("A forged CNN screenshot says a Nigerian senator was arrested at a UK "
           "airport with $15 million in a suitcase.",
           "Dubawa", "politics", "false"),
    Record("A manipulated video makes it look like a religious cleric endorsed a "
           "particular political party from the pulpit.",
           "FactCheckHub", "politics", "misleading"),
    Record("A fake Aliko Dangote broadcast asks people to invest in a crypto scheme "
           "that doubles their money in 48 hours.",
           "PesaCheck", "scam", "false"),
    Record("A quote falsely attributed to the CBN Governor says citizens should keep "
           "all their money at home instead of in banks.",
           "AfricaCheck", "economy", "false"),

    # --- Online scams / giveaways ------------------------------------------
    Record("Congratulations! Your MTN line has been selected to win N2 million; "
           "click this link and enter your card PIN to claim.",
           "FactCheckHub", "scam", "false"),
    Record("The federal government is sharing N30,000 palliative to every phone "
           "number, just forward this message to ten people to receive yours.",
           "Dubawa", "scam", "false"),
    Record("A WhatsApp message says NNPC is recruiting 50,000 workers with no "
           "interview; pay N7,500 for a guaranteed appointment letter.",
           "FactCheckHub", "scam", "false"),
    Record("Dangote Foundation is giving free bags of rice and N20,000 to everyone "
           "who fills this Google form with their BVN.",
           "AfricaCheck", "scam", "false"),
    Record("Click to confirm your NIN or your SIM will be blocked in 24 hours, a "
           "phishing message impersonating NIMC warns.",
           "Dubawa", "scam", "false"),

    # --- Miscellaneous rumours ---------------------------------------------
    Record("A viral post claims that a full moon this week will emit radiation, so "
           "everyone must switch off their phones overnight.",
           "AfricaCheck", "science", "false"),
    Record("Adding a spoon of table salt to your phone battery makes it charge twice "
           "as fast, a trending video claims.",
           "FactCheckHub", "science", "false"),
    Record("NASA has confirmed six days of total darkness over Nigeria next month, "
           "according to a widely shared broadcast.",
           "Dubawa", "science", "false"),
    Record("A message insists that boiling plastic sachet water releases a chemical "
           "that instantly causes cancer after one sip.",
           "AfricaCheck", "health", "misleading"),
    Record("Schools nationwide have been ordered to close indefinitely from tomorrow, "
           "a fake ministry memo announces.",
           "FactCheckHub", "politics", "false"),
    Record("A recycled photo of a flood abroad is shared as the current situation in "
           "a Nigerian state to solicit donations.",
           "Dubawa", "disaster", "misleading"),
    Record("Viral claim: eating bananas and drinking Sprite together forms a poison "
           "that has killed several students.",
           "AfricaCheck", "health", "false"),
    Record("A trending message says the government has banned the use of generators "
           "in all homes starting next week.",
           "FactCheckHub", "politics", "false"),
    Record("A manipulated headline claims a popular Nollywood actor has died, urging "
           "fans to share for 'RIP' before the family confirmed anything.",
           "Dubawa", "celebrity", "false"),
    Record("A message claims that dialling a particular USSD code lets network staff "
           "clone your phone and empty your bank account instantly.",
           "FactCheckHub", "scam", "misleading"),
    Record("A viral flyer says JAMB has cancelled this year's UTME and all candidates "
           "will be admitted automatically.",
           "Dubawa", "education", "false"),
    Record("A doctored WHO map claims Lagos is the most polluted city on earth, "
           "beating every other global megacity.",
           "AfricaCheck", "science", "misleading"),
    Record("A forwarded note warns that drinking cold water immediately after hot "
           "food causes instant stroke in young Nigerians.",
           "FactCheckHub", "health", "false"),
    Record("A viral audio claims the government has approved a law making it illegal "
           "to speak any indigenous language in public offices.",
           "Dubawa", "politics", "false"),
    Record("A screenshot claims that a foreign airline is offering free flights out "
           "of Nigeria to the first 10,000 people who register.",
           "PesaCheck", "scam", "false"),
    Record("A message says the total lunar eclipse will make all pregnant women give "
           "birth prematurely unless they stay indoors.",
           "AfricaCheck", "science", "false"),
    Record("A recycled 2016 fuel-queue photo is shared as proof of a brand-new "
           "petrol scarcity gripping the whole country today.",
           "Dubawa", "economy", "misleading"),
    Record("A fake NCDC alert says a new airborne disease is spreading through "
           "mobile phone screens and people should stop using WhatsApp.",
           "FactCheckHub", "health", "false"),
    Record("A viral claim says the ozone layer above West Africa has completely "
           "disappeared and sunlight will now burn human skin in seconds.",
           "AfricaCheck", "science", "false"),
    Record("A forwarded message asserts that a particular brand of noodles contains "
           "plastic wax that the body can never digest.",
           "Dubawa", "health", "misleading"),
]


# ---------------------------------------------------------------------------
# CREDIBLE  (verdict = true)
# ---------------------------------------------------------------------------
CREDIBLE: List[Record] = [
    Record("The Central Bank of Nigeria extended the deadline for swapping old naira "
           "notes following a Supreme Court ruling in 2023.",
           "TheCable", "economy", "true"),
    Record("INEC introduced the Bimodal Voter Accreditation System (BVAS) to verify "
           "voters using fingerprints and facial recognition at polling units.",
           "AfricaCheck", "elections", "true"),
    Record("The World Health Organization recommends regular handwashing with soap "
           "and water to reduce the spread of respiratory infections.",
           "Dubawa", "health", "true"),
    Record("Nigeria officially removed the petrol subsidy in 2023, leading to an "
           "increase in pump prices across the country.",
           "TheCable", "economy", "true"),
    Record("NAFDAC is the Nigerian agency responsible for regulating and approving "
           "food, drugs and cosmetics before they are sold to the public.",
           "FactCheckHub", "health", "true"),
    Record("The National Bureau of Statistics publishes Nigeria's official inflation "
           "figures on a monthly basis.",
           "PesaCheck", "economy", "true"),
    Record("Lassa fever is spread mainly through contact with food or items "
           "contaminated by the urine or faeces of infected rats.",
           "NCDC", "health", "true"),
    Record("The Nigeria Centre for Disease Control advises people with symptoms of an "
           "infectious disease to call the toll-free line and isolate.",
           "Dubawa", "health", "true"),
    Record("Nigeria's 2023 presidential election was conducted across 36 states and "
           "the Federal Capital Territory in February 2023.",
           "AfricaCheck", "elections", "true"),
    Record("The Central Bank of Nigeria set the eNaira as the country's official "
           "central bank digital currency, launched in October 2021.",
           "TheCable", "economy", "true"),
    Record("Malaria is transmitted to humans through the bite of an infected female "
           "Anopheles mosquito, according to the WHO.",
           "AfricaCheck", "health", "true"),
    Record("The Nigerian government introduced the National Identification Number "
           "(NIN) as a unique identifier managed by NIMC.",
           "FactCheckHub", "politics", "true"),
    Record("JAMB conducts the Unified Tertiary Matriculation Examination (UTME) as a "
           "computer-based test for admission into Nigerian universities.",
           "Dubawa", "education", "true"),
    Record("Routine childhood immunisation protects against diseases such as measles, "
           "polio, tuberculosis and diphtheria.",
           "AfricaCheck", "health", "true"),
    Record("The Lagos State Government operates the Bus Rapid Transit (BRT) system to "
           "ease commuting along major corridors in the city.",
           "TheCable", "transport", "true"),
    Record("The Dangote Refinery in Lekki, Lagos, is one of the largest single-train "
           "refineries built to process crude oil in Africa.",
           "PesaCheck", "economy", "true"),
    Record("COVID-19 vaccines authorised for use in Nigeria went through clinical "
           "trials and regulatory review before approval by NAFDAC.",
           "Dubawa", "health", "true"),
    Record("The naira is the official currency of Nigeria and is issued solely by the "
           "Central Bank of Nigeria.",
           "AfricaCheck", "economy", "true"),
    Record("Voter registration in Nigeria is handled by INEC through the Continuous "
           "Voter Registration exercise.",
           "FactCheckHub", "elections", "true"),
    Record("Cholera outbreaks in Nigeria are commonly linked to contaminated drinking "
           "water and poor sanitation, the NCDC says.",
           "NCDC", "health", "true"),
    Record("The Second Niger Bridge connecting Anambra and Delta states was "
           "commissioned to ease traffic across the River Niger.",
           "TheCable", "infrastructure", "true"),
    Record("Nigeria's House of Representatives and Senate together make up the "
           "bicameral National Assembly.",
           "AfricaCheck", "politics", "true"),
    Record("Tuberculosis is a treatable bacterial disease, and free treatment is "
           "available at accredited health facilities in Nigeria.",
           "Dubawa", "health", "true"),
    Record("The Federal Inland Revenue Service is responsible for assessing and "
           "collecting federal taxes such as company income tax and VAT.",
           "PesaCheck", "economy", "true"),
    Record("Insecticide-treated mosquito nets are distributed in malaria-endemic "
           "parts of Nigeria to reduce transmission.",
           "AfricaCheck", "health", "true"),
    Record("The Nigerian Meteorological Agency issues seasonal rainfall predictions "
           "to guide farmers and disaster planning.",
           "FactCheckHub", "science", "true"),
    Record("A citizen must be at least 18 years old to be eligible to register and "
           "vote in Nigerian elections.",
           "AfricaCheck", "elections", "true"),
    Record("The Nigerian Communications Commission directed the linkage of SIM cards "
           "to the National Identification Number.",
           "Dubawa", "politics", "true"),
    Record("Yellow fever is a vaccine-preventable disease, and a single dose offers "
           "long-lasting protection, the WHO states.",
           "AfricaCheck", "health", "true"),
    Record("Nigeria's fiscal budget is presented by the President and passed into law "
           "by the National Assembly each year.",
           "TheCable", "politics", "true"),
    Record("The Petroleum Industry Act was signed into law in 2021 to reform the "
           "governance of Nigeria's oil and gas sector.",
           "PesaCheck", "economy", "true"),
    Record("Regular exercise and a balanced diet help reduce the risk of type 2 "
           "diabetes and heart disease.",
           "Dubawa", "health", "true"),
    Record("The Federal Road Safety Corps enforces the use of seat belts and speed "
           "limits on Nigerian highways.",
           "AfricaCheck", "transport", "true"),
    Record("Nigeria's official statistics agency reported the country's population "
           "estimate exceeds 200 million people.",
           "FactCheckHub", "economy", "true"),
    Record("The National Youth Service Corps (NYSC) requires eligible graduates to "
           "complete one year of national service.",
           "TheCable", "education", "true"),
    Record("Antibiotics are effective against bacterial infections but not against "
           "viral illnesses such as the common cold.",
           "AfricaCheck", "health", "true"),
    Record("The Economic and Financial Crimes Commission is Nigeria's agency for "
           "investigating financial crimes such as fraud and money laundering.",
           "Dubawa", "politics", "true"),
    Record("Exclusive breastfeeding for the first six months is recommended by the "
           "WHO for the health of infants.",
           "AfricaCheck", "health", "true"),
    Record("The Nigerian stock market is operated by the Nigerian Exchange Group, "
           "formerly the Nigerian Stock Exchange.",
           "PesaCheck", "economy", "true"),
    Record("Lagos is Nigeria's most populous city and its main commercial centre, "
           "while Abuja is the federal capital.",
           "AfricaCheck", "geography", "true"),
    Record("The Nigeria Police Force operates emergency response lines that citizens "
           "can call to report crimes.",
           "Dubawa", "security", "true"),
    Record("The Universal Basic Education programme provides free and compulsory "
           "primary and junior secondary education in Nigeria.",
           "TheCable", "education", "true"),
    Record("Hepatitis B can be prevented through vaccination, which is part of "
           "Nigeria's routine immunisation schedule.",
           "AfricaCheck", "health", "true"),
    Record("The Central Bank of Nigeria uses the Monetary Policy Rate as its main "
           "tool for controlling inflation and interest rates.",
           "PesaCheck", "economy", "true"),
    Record("INEC uses the IReV portal to upload scanned copies of polling-unit result "
           "sheets for public viewing.",
           "FactCheckHub", "elections", "true"),
    Record("Drinking clean, treated water helps prevent water-borne diseases such as "
           "typhoid and cholera.",
           "Dubawa", "health", "true"),
    Record("The Nigerian passport is issued by the Nigeria Immigration Service to "
           "citizens for international travel.",
           "AfricaCheck", "politics", "true"),
    Record("Farmers in northern Nigeria commonly cultivate crops such as millet, "
           "sorghum, maize and groundnut.",
           "PesaCheck", "agriculture", "true"),
    Record("The Sustainable Development Goals include targets on ending poverty, "
           "improving health and expanding access to education by 2030.",
           "AfricaCheck", "development", "true"),
    Record("Nigeria gained independence from British colonial rule on 1 October 1960.",
           "TheCable", "history", "true"),
    Record("The National Health Insurance Authority coordinates health insurance "
           "coverage for Nigerians under a 2022 law.",
           "Dubawa", "health", "true"),
    Record("The Nigerian Electricity Regulatory Commission regulates tariffs and "
           "standards in the country's power sector.",
           "PesaCheck", "economy", "true"),
    Record("Handwashing with soap before eating and after using the toilet reduces "
           "the spread of diarrhoeal diseases.",
           "AfricaCheck", "health", "true"),
    Record("The West African Examinations Council conducts the WASSCE for secondary "
           "school students across the region.",
           "FactCheckHub", "education", "true"),
    Record("The River Niger and River Benue are the two largest rivers in Nigeria and "
           "meet at Lokoja.",
           "AfricaCheck", "geography", "true"),
    Record("Sickle cell disease is an inherited blood disorder, and genotype testing "
           "helps couples understand their risk.",
           "Dubawa", "health", "true"),
    Record("The Federal Ministry of Health coordinates national responses to disease "
           "outbreaks alongside the NCDC.",
           "AfricaCheck", "health", "true"),
    Record("The naira redesign policy in 2022 introduced new designs for the N200, "
           "N500 and N1,000 notes.",
           "TheCable", "economy", "true"),
    Record("Regular antenatal care improves outcomes for mothers and babies during "
           "pregnancy, health authorities advise.",
           "AfricaCheck", "health", "true"),
    Record("The Nigerian Army, Navy and Air Force make up the country's armed forces "
           "under the Ministry of Defence.",
           "Dubawa", "security", "true"),
]


def build() -> List[Record]:
    records = MISINFO + CREDIBLE
    # basic integrity checks
    seen = set()
    for r in records:
        assert r.verdict in {"false", "misleading", "true"}, r.verdict
        key = r.text.strip().lower()
        assert key not in seen, f"duplicate text: {r.text[:60]}"
        seen.add(key)
    return records


def write_csv(path: str = OUT_CSV) -> str:
    records = build()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id", "text", "source", "category", "verdict", "label"])
        for i, r in enumerate(records, start=1):
            writer.writerow([i, r.text, r.source, r.category, r.verdict, r.label])
    return path


if __name__ == "__main__":
    recs = build()
    path = write_csv()
    n_mis = sum(1 for r in recs if r.label == 1)
    n_cred = sum(1 for r in recs if r.label == 0)
    print(f"Wrote {len(recs)} records -> {path}")
    print(f"  misinformation (1): {n_mis}")
    print(f"  credible       (0): {n_cred}")
