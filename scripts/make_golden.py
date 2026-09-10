"""Stratified 200-row golden draft (labels overwritten by hand). --help/--sample-n/--seed/--out."""
import argparse, re, pandas as pd, numpy as np
INTENTS = ["setup_howto","battery_performance","software_update","account_icloud","hardware_damage_repair","warranty_applecare","app_store_itunes","connectivity","billing_refund","other_offtopic"]
KW = {"setup_howto":["setup","pair","how do"],"battery_performance":["battery","drain","overheat","slow"],"software_update":["update","ios ","upgrade"],"account_icloud":["icloud","apple id","login","password"],"hardware_damage_repair":["crack","broken","repair","screen"],"warranty_applecare":["warranty","applecare","coverage"],"app_store_itunes":["app store","subscription","itunes"],"connectivity":["wifi","bluetooth","cellular","airdrop"],"billing_refund":["refund","charg","receipt","payment"]}
def weak(t):
    t = str(t).lower()
    for k, words in KW.items():
        if any(w in t for w in words): return k
    return "other_offtopic"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/threads_sample.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/golden.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.inp)
    rng = np.random.RandomState(a.seed)
    df["_w"] = df.customer_text.apply(weak)
    pick = []
    per = 15
    for k in INTENTS:
        sub = df[df._w == k]
        pick += list(sub.sample(n=min(per, len(sub)), random_state=a.seed).index)
    rest = df.drop(index=pick)
    need_rand = 30
    pick += list(rest.sample(n=min(need_rand, len(rest)), random_state=a.seed+1).index)
    rest2 = df.drop(index=pick)
    # adversarial: short, caps, sarcasm, multikeyword
    adv = rest2[(rest2.customer_text.str.len() < 60) | (rest2.customer_text.str.contains(r"[!?]{2,}|\bwhatever\b|sarcasm|ASDF|\bwtf\b", case=False, na=False))].head(20)
    if len(adv) < 20: adv = rest2.sample(n=min(20, len(rest2)), random_state=a.seed+2)
    pick += list(adv.index)
    g = df.loc[pick].head(a.n).copy()
    g["intent_gold"] = g["_w"]; g["reply_reference"] = ""; g["escalate_gold"] = False
    g["escalate_reason_gold"] = "OK:needs-hand-label"; g["annotator"] = ""; g["notes"] = "draft-weak-label-overwrite-me"
    out = g[["msg_id","brand","customer_text","parent_brand_text","intent_gold","reply_reference","escalate_gold","escalate_reason_gold","annotator","notes"]].rename(columns={"msg_id":"id","parent_brand_text":"context_parent_text"})
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out} rows={len(out)} (hand-label ALL rows before eval)")
if __name__ == "__main__": main()
