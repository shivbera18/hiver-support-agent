"""B1: TF-IDF+LogReg on weak labels, NN reply, threshold escalate. --help/--sample-n/--seed/--out."""
import argparse, json, re, pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
INTENTS = ["setup_howto","battery_performance","software_update","account_icloud","hardware_damage_repair","warranty_applecare","app_store_itunes","connectivity","billing_refund","other_offtopic"]
KW = {"setup_howto":["setup","pair","how do"],"battery_performance":["battery","drain","overheat","slow"],"software_update":["update","ios","upgrade"],"account_icloud":["icloud","apple id","login","password"],"hardware_damage_repair":["crack","broken","repair","screen"],"warranty_applecare":["warranty","applecare","coverage"],"app_store_itunes":["app store","subscription","itunes"],"connectivity":["wifi","bluetooth","cellular","airdrop"],"billing_refund":["refund","charg","receipt","payment"]}
RISK = re.compile(r"refund|chargeback|lawyer|sue|broken|crack|shatter|stolen|safety|threat|kill|self.?harm|dm me|password|\bssn\b|card number", re.I)
CANNED = "Thanks for reaching out \u2014 could you share your device model and iOS version plus what you\u2019ve tried so far?"
def weak(t):
    t = str(t).lower()
    for k, words in KW.items():
        if any(w in t for w in words): return k
    return "other_offtopic"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/threads_sample.csv")
    ap.add_argument("--golden", default="data/golden.csv")
    ap.add_argument("--out", default="results/predictions_baseline_tfidf.jsonl")
    ap.add_argument("--sample-n", type=int, default=None); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = pd.read_csv(a.inp)
    g = pd.read_csv(a.golden)
    gids = set(g["id"].astype(str))
    df = df[~df.msg_id.astype(str).isin(gids)]
    df = df.sort_values("created_at")
    n = min(20000, len(df)); tr = df.head(int(n*0.8))
    y = tr.customer_text.apply(weak)
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1,2))
    X = vec.fit_transform(tr.customer_text.astype(str))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(X, y)
    tr_rep = tr[tr.brand_reply_text.astype(str).str.len() > 10].copy()
    try:
        _raw = pd.read_csv(a.inp.replace("threads_sample.csv", "twcs_raw.csv") if "threads_sample" in a.inp else "data/twcs_raw.csv", usecols=["author_id", "inbound", "text"], low_memory=False)
        _ob = _raw[(_raw.inbound == False) & (_raw.author_id.astype(str).str.lower() == "applesupport")].copy()
        _ob = _ob.rename(columns={"text": "brand_reply_text"}); _ob["customer_text"] = _ob.brand_reply_text
        tr_rep = pd.concat([tr_rep, _ob.head(8000)], ignore_index=True)
        print(f"reply pool augmented: total={len(tr_rep)}")
    except Exception as e: print("augment skipped:", e)
    Xr = vec.transform(tr_rep.customer_text.astype(str))
    rep_texts = tr_rep.brand_reply_text.tolist()
    if a.sample_n: g = g.head(a.sample_n)
    with open(a.out, "w", encoding="utf-8") as f:
        for _, r in g.iterrows():
            t = str(r["customer_text"])
            xv = vec.transform([t])
            proba = clf.predict_proba(xv)[0]; pred = clf.classes_[proba.argmax()]; conf = float(proba.max())
            sims = cosine_similarity(xv, Xr)[0] if len(rep_texts) else np.array([0])
            reply = rep_texts[int(sims.argmax())] if len(rep_texts) and sims.max() >= 0.15 else CANNED
            risk = bool(RISK.search(t))
            esc = (conf < 0.60) or risk
            reason = f"RISK:pattern-match" if risk else (f"LOWCONF:{conf:.2f}" if conf < 0.60 else "OK:auto-resolvable")
            f.write(json.dumps({"id": str(r["id"]), "intent": str(pred), "reply": str(reply)[:280], "escalate": bool(esc), "reason": reason, "conf": conf}) + "\n")
    print(f"wrote {a.out}")
if __name__ == "__main__": main()
