"""Main hybrid RAG agent: classify/draft_reply/decide. --help/--sample-n/--seed/--out. --smoke for 1-row check."""
import argparse, json, re, os, pandas as pd, numpy as np
INTENTS = ["setup_howto","battery_performance","software_update","account_icloud","hardware_damage_repair","warranty_applecare","app_store_itunes","connectivity","billing_refund","other_offtopic"]
RISK = re.compile(r"refund|charg|payment|bill|receipt|chargeback|lawyer|sue|broken|crack|shatter|stolen|safety|threat|kill|self.?harm|dm me|password|\bssn\b|card number|scam|fraud|phish|expir|unauthori|suspend|\b\d{3}-\d{2}-\d{4}\b", re.I)
PII = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|password\s*[:=]\s*\S+|\b\d{13,19}\b", re.I)
CANNED = "Thanks for reaching out \u2014 could you share your device model and iOS version plus what you\u2019ve tried so far?"
THRESH = float(__import__('os').environ.get("AGENT_THRESH", "0.55"))
try:
    from dotenv import load_dotenv; load_dotenv(".env", override=True)
except Exception: pass
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
try:
    from pydantic import BaseModel, confloat, ValidationError
    from typing import Literal
    class IntentOut(BaseModel):
        intent: Literal["setup_howto","battery_performance","software_update","account_icloud","hardware_damage_repair","warranty_applecare","app_store_itunes","connectivity","billing_refund","other_offtopic"]
        confidence: confloat(ge=0.0, le=1.0)
    _HAVE_PYDANTIC = True
except Exception: _HAVE_PYDANTIC = False
_model = {}
_USAGE = {"calls": 0, "ok": 0, "prompt_tokens": 0, "completion_tokens": 0, "fallback_rows": 0}
def _log_call(stage, model, ok, err="", usage=None):
    import json as _j
    _USAGE["calls"] += 1
    if ok: _USAGE["ok"] += 1
    pt = (usage.prompt_tokens if usage and hasattr(usage, "prompt_tokens") else 0) or 0
    ct = (usage.completion_tokens if usage and hasattr(usage, "completion_tokens") else 0) or 0
    _USAGE["prompt_tokens"] += pt; _USAGE["completion_tokens"] += ct
    print(f"LLM {stage} {model} {'OK' if ok else 'FAIL:' + str(err)[:100]} tok={pt}/{ct}", flush=True)
_WEAK_KW = {"setup_howto": ["setup", "pair", "how do"], "battery_performance": ["battery", "drain", "overheat", "slow"], "software_update": ["update", "ios", "upgrade"], "account_icloud": ["icloud", "apple id", "login", "password"], "hardware_damage_repair": ["crack", "broken", "repair", "screen"], "warranty_applecare": ["warranty", "applecare", "coverage"], "app_store_itunes": ["app store", "subscription", "itunes"], "connectivity": ["wifi", "bluetooth", "cellular", "airdrop"], "billing_refund": ["refund", "charg", "receipt", "payment"]}
def _weak(t):
    t = str(t).lower()
    for k, words in _WEAK_KW.items():
        if any(w in t for w in words):
            return k
    return "other_offtopic"
def _local_clf(pool_texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    y = [_weak(t) for t in pool_texts]
    vec=TfidfVectorizer(max_features=20000, ngram_range=(1,2)); X=vec.fit_transform([str(t) for t in pool_texts])
    clf=LogisticRegression(max_iter=1000, class_weight="balanced").fit(X,y)
    return vec, clf
def _retriever(pool):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    rep = pool[pool.brand_reply_text.astype(str).str.len()>10].copy()
    try:
        from sentence_transformers import SentenceTransformer
        enc = SentenceTransformer("all-MiniLM-L6-v2")
        E = enc.encode(rep.customer_text.tolist(), show_progress_bar=False, normalize_embeddings=True)
        return ("sbert", enc, rep, E)
    except Exception as e:
        print("sbert unavailable, TF-IDF retrieval:", e)
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
        E = vec.fit_transform(rep.customer_text.astype(str))
        return ("tfidf", vec, rep, E)
def _gpt_classify(text, context):
    try:
        from openai import OpenAI
        c = OpenAI()
        prompt = open("prompts/intent_prompt.txt").read()
        msg = f"CURRENT: {text}\nPARENT: {context}"
        r = c.chat.completions.create(model=OPENAI_MODEL, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": msg}],
            max_completion_tokens=600)
        d = IntentOut.model_validate_json(r.choices[0].message.content)
        _log_call("classify", OPENAI_MODEL, True, usage=r.usage)
        return str(d.intent), float(d.confidence)
    except Exception as e:
        _log_call("classify", OPENAI_MODEL, False, str(e)); return None

def _gretry(fn, tries=3):
    import time
    for i in range(tries):
        try: return fn()
        except Exception as e:
            print(f"gemini try {i+1} failed:", str(e)[:100])
            time.sleep(2 * (i + 1))
    return None

def _gemini_classify(text, context):
    try:
        from google import genai
        g = genai.Client()
        prompt = open("prompts/intent_prompt.txt").read()
        tail = 'Reply with JSON only, e.g. {"intent": "billing_refund", "confidence": 0.9}'
        msg = "CURRENT: " + str(text) + "\nPARENT: " + str(context) + "\n" + tail
        r = _gretry(lambda: g.models.generate_content(model=GEMINI_MODEL, contents=prompt + "\n" + msg))
        if r is None: return None
        t = re.sub(r"^```(json)?|```$", "", r.text.strip()).strip()
        d = IntentOut.model_validate_json(t)
        _log_call("classify", GEMINI_MODEL, True)
        return str(d.intent), float(d.confidence)
    except Exception as e:
        _log_call("classify", GEMINI_MODEL, False, str(e)); return None

def classify(text, context=""):
    votes = []
    if os.environ.get("OPENAI_API_KEY") and _HAVE_PYDANTIC:
        v = _gpt_classify(text, context)
        if v: votes.append(v)
    # cascade: ask gemini only when gpt is unsure or silent (saves ~70% of calls)
    if os.environ.get("GEMINI_API_KEY") and os.environ.get("GEMINI_ENABLE") and _HAVE_PYDANTIC and (not votes or votes[0][1] < 0.7):
        v = _gemini_classify(text, context)
        if v: votes.append(v)
    if len(votes) == 2 and votes[0][0] == votes[1][0]:
        pred, conf = votes[0][0], max(votes[0][1], votes[1][1])
    elif votes:
        pred, conf = max(votes, key=lambda v: v[1])
    else:
        if os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            _USAGE["fallback_rows"] += 1
            print("ROW FALLBACK: no llm votes, tfidf classify")
        vec, clf = _model["clf"]; p = clf.predict_proba(vec.transform([text]))[0]
        pred, conf = str(clf.classes_[p.argmax()]), float(p.max())
    t = str(text)
    shortened = t.strip().lower() in ("see pic", "pic attached", "itag?")
    if sum(ord(c) > 127 for c in t) / max(len(t), 1) > 0.3 or len(t.split()) <= 3 or shortened:
        return "other_offtopic", 0.0
    if re.search(r"charged|refund|payment|receipt|overcharg", t, re.I) and pred in ("app_store_itunes", "other_offtopic"):
        return "billing_refund", max(conf, 0.65)
    return pred, conf
def _gpt_reply(text, context, intent, hist):
    from openai import OpenAI
    c = OpenAI()
    prompt = open("prompts/reply_prompt.txt").read()
    msg = f"INTENT: {intent}\nCUSTOMER: {text}\nCONTEXT: {context}\nHISTORICAL RESOLUTIONS:\n{hist}"
    r = c.chat.completions.create(model=OPENAI_MODEL,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": msg}],
        max_completion_tokens=1000)
    _log_call("reply", OPENAI_MODEL, True, usage=r.usage)
    return r.choices[0].message.content.strip()[:280]

def _gemini_reply(text, context, intent, hist):
    from google import genai
    g = genai.Client()
    prompt = open("prompts/reply_prompt.txt").read()
    msg = f"INTENT: {intent}\nCUSTOMER: {text}\nCONTEXT: {context}\nHISTORICAL RESOLUTIONS:\n{hist}\nReply with the reply text only."
    r = _gretry(lambda: g.models.generate_content(model=GEMINI_MODEL, contents=prompt + "\n" + msg))
    if r is None: raise RuntimeError("gemini unavailable")
    return r.text.strip()[:280]

def draft_reply(text, context, intent, retrieved):
    hist = "\n".join(f"- {r}" for r in retrieved[:3]) if retrieved else "- (no history)"
    if os.environ.get("OPENAI_API_KEY"):
        try: return _gpt_reply(text, context, intent, hist)
        except Exception as e: _log_call("reply", OPENAI_MODEL, False, str(e))
    if os.environ.get("GEMINI_API_KEY") and os.environ.get("GEMINI_ENABLE"):
        try:
            out = _gemini_reply(text, context, intent, hist)
            _log_call("reply", GEMINI_MODEL, True)
            return out
        except Exception as e: _log_call("reply", GEMINI_MODEL, False, str(e))
    return str(retrieved[0])[:280] if retrieved else CANNED
def decide(text, intent, conf):
    t = str(text); non_en = sum(ord(ch) > 127 for ch in t) / max(len(t), 1) > 0.3
    if intent == "billing_refund":
        return True, "RISK:billing-intent"
    if RISK.search(t): return True, "RISK:pattern-match"
    if non_en: return True, "RISK:language_unsupported"
    if len(t.split()) <= 3 or t.strip().lower() in ("see pic","pic attached","itag?"): return True, "MISSING-INFO:too-short"
    if conf < THRESH: return True, f"LOWCONF:{conf:.2f}"
    if intent == "other_offtopic" and re.search(r"complaint|angry|worst|terrible|broken|refund", t, re.I): return True, "RISK:offtopic-complaint"
    return False, "OK:auto-resolvable-howto"
def _retrieve(text, R, top=3):
    kind, enc, rep, E = R
    if kind == "sbert":
        q = enc.encode([text], normalize_embeddings=True); s = (q @ E.T)[0]
    else:
        from sklearn.metrics.pairwise import cosine_similarity
        s = cosine_similarity(enc.transform([text]), E)[0]
    idx = np.argsort(-s)[:top]
    return rep.brand_reply_text.iloc[idx].tolist()
def run(inp, golden, out, sample_n=None, seed=42):
    df = pd.read_csv(inp); g = pd.read_csv(golden)
    gids = set(g["id"].astype(str)); pool = df[~df.msg_id.astype(str).isin(gids)]
    pool = pool.sort_values("created_at").head(min(20000, len(pool)))
    try:
        _raw = pd.read_csv("data/twcs_raw.csv", usecols=["author_id", "inbound", "text"], low_memory=False)
        _ob = _raw[(_raw.inbound == False) & (_raw.author_id.astype(str).str.lower() == "applesupport")].copy()
        _ob = _ob.rename(columns={"text": "brand_reply_text"}); _ob["customer_text"] = _ob.brand_reply_text
        pool = pd.concat([pool, _ob.head(8000)], ignore_index=True); print(f"reply pool augmented: total={len(pool)}")
    except Exception as e: print("augment skipped:", e)
    _model["clf"] = _local_clf(pool.customer_text.tolist())
    R = _retriever(pool)
    R_by_intent = {}
    pool["_w"] = pool.customer_text.apply(_weak)
    for k in INTENTS:
        sub = pool[pool._w == k]
        R_by_intent[k] = _retriever(sub) if len(sub) > 5 else R
    print("intent-filtered retrieval on")
    if sample_n: g = g.head(sample_n)
    rows = []
    for _, r in g.iterrows():
        t = str(r["customer_text"]); ctx = str(r.get("context_parent_text", "") or "")
        intent, conf = classify(t, ctx)
        R_use = R_by_intent.get(intent, R) if R_by_intent else R
        ret = [] if __import__("os").environ.get("AGENT_NORETRIEVAL") else _retrieve(t, R_use)
        reply = draft_reply(t, ctx, intent, ret)
        if PII.search(reply): reply = PII.sub("[REDACTED]", reply); esc, reason = True, "SAFETY:pii-redacted"
        else: esc, reason = decide(t, intent, conf)
        if "refund" in reply.lower() and "eligib" not in reply.lower() and "cannot promise" not in reply.lower():
            esc, reason = True, "RISK:refund-promise-check"
        rows.append({"id": str(r["id"]), "intent": intent, "reply": reply, "escalate": bool(esc), "reason": reason, "conf": float(conf)})
    import json as _j
    floor = float(os.environ.get("LLM_MIN_SUCCESS", "0.8"))
    rate = _USAGE["ok"] / max(_USAGE["calls"], 1)
    print(f"LLM SUMMARY calls={_USAGE['calls']} ok={_USAGE['ok']} rate={rate:.2f} fallback_rows={_USAGE['fallback_rows']} tok_in={_USAGE['prompt_tokens']} tok_out={_USAGE['completion_tokens']}")
    _j.dump(_USAGE, open("results/llm_usage.json", "w"), indent=2)
    if _USAGE["calls"] > 0 and rate < floor:
        raise SystemExit(f"ABORT: llm success {rate:.2f} < {floor} -- refusing to label fallback output as llm results")
    pd.DataFrame(rows).to_json(out, orient="records", lines=True, force_ascii=False)
    print(f"wrote {out} rows={len(rows)} thresh={THRESH} llm={'on' if os.environ.get('OPENAI_API_KEY') else 'fallback'}")
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/threads_sample.csv"); ap.add_argument("--golden", default="data/golden.csv")
    ap.add_argument("--out", default="results/predictions_main.jsonl"); ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        _model["clf"] = _local_clf(["battery drains fast","refund please"])
        print(classify("my iPhone battery drains in an hour"))
        print(draft_reply("my iPhone battery drains", "", "battery_performance", ["Try Settings > Battery to check usage. What model/iOS are you on?"]))
        print(decide("give me a refund now!", "billing_refund", 0.9)); return
    run(a.inp, a.golden, a.out, a.sample_n, a.seed)
if __name__ == "__main__": main()
