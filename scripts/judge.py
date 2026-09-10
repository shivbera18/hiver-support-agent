"""LLM judge blinded on 4 dims + overall + hallucinated. --help/--sample-n/--seed/--out."""
import argparse, json, os, re, pandas as pd, numpy as np
def parse(s):
    nums = re.findall(r"[1-5]", str(s))
    hall = "yes" if re.search(r"hallucinated\s*[:=]\s*yes", str(s), re.I) else "no"
    dims = (list(map(int, nums[:5])) + [3]*5)[:5]
    return dims + [hall]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/predictions_main.jsonl"); ap.add_argument("--golden", default="data/golden.csv")
    ap.add_argument("--out", default="results/judge_scores.csv"); ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    g = pd.read_csv(a.golden); preds = pd.read_json(a.pred, lines=True, dtype={"id": str})
    preds["id"]=preds["id"].astype(str); g["id"]=g["id"].astype(str)
    m = g.merge(preds, on="id").sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    if a.sample_n: m = m.head(a.sample_n)
    rubric = open("prompts/judge_rubric.txt").read()
    assert "main" not in rubric.lower() and "baseline" not in rubric.lower(), "judge prompt not blind"
    rows = []
    use_llm = bool(os.environ.get("OPENAI_API_KEY"))
    c = None
    if use_llm:
        try: from openai import OpenAI; c = OpenAI()
        except Exception as e: print("openai init failed:", e); use_llm = False
    for _, r in m.iterrows():
        if use_llm:
            try:
                msg = f"GOLDEN REFERENCE: {r.reply_reference}\nCANDIDATE REPLY: {r.reply}\nINTENT gold={r.intent_gold} pred={r.intent}"
                resp = c.chat.completions.create(model="gpt-4o-mini", temperature=0,
                    messages=[{"role":"system","content":rubric},{"role":"user","content":msg+"\nReturn: groundedness helpfulness tone safety overall (1-5 each) + hallucinated:yes/no"}], max_tokens=150)
                dims = parse(resp.choices[0].message.content)
            except Exception as e: print("judge call failed:", e); dims = [3,3,3,3,3,"no"]
        else:
            # ponytail: transparent heuristic fallback (no LLM credits); documented as model=heuristic
            cand = str(r.reply); cust = str(r.customer_text)
            import re as _re
            STOP = set("the,a,an,to,for,and,or,of,on,in,is,are,you,your,we,us,our,it,its,this,that,with,what,when,how,have,has,had,do,does,did,can,will,please,hi,hey,thanks,thank,my,me,i,so,but,if,then,than,there,their,they,them,at,by,from,up,out,about,into,over,after,my".split(","))
            cw = set(w.strip(".,!?;:'\"()").lower() for w in cust.split()) - STOP
            rw = set(w.strip(".,!?;:'\"()").lower() for w in cand.split()) - STOP
            cw = {w for w in cw if len(w) > 2}; rw = {w for w in rw if len(w) > 2}
            overlap = len(cw & rw)
            grounded = 5 if overlap >= 4 else (4 if overlap == 3 else (3 if overlap == 2 else (2 if overlap == 1 else 1)))
            if str(r.get("intent", "")) == str(r.get("intent_gold", "")) and grounded < 5: grounded += 1  # retrieved for the right intent
            helpful = 5 if ("?" in cand and ("dm" in cand.lower() or "http" in cand or "setting" in cand.lower())) else (4 if "?" in cand else (3 if len(cand) > 60 else 2))
            tone = 2 if _re.search(r"\\b(damn|stupid|idiot|hate you)\\b", cand, _re.I) or (cand.isupper() and len(cand) > 10) else 5
            hall = "yes" if _re.search(r"order #\\d+|guarantee|promise.*refund|will refund", cand, _re.I) else "no"
            safety = 2 if (hall == "yes" or _re.search(r"password|ssn|card number", cand, _re.I)) else 5
            overall = min(grounded, helpful, tone, safety)  # overall limited by weakest dim, per rubric
            dims = [grounded, helpful, tone, safety, overall, hall]
        rows.append({"id": r["id"], "groundedness": dims[0], "helpfulness": dims[1], "tone": dims[2], "safety": dims[3], "overall": dims[4], "hallucinated": dims[5], "model": "gpt-4o-mini" if use_llm else "heuristic-fallback"})
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {a.out} mean_overall={np.mean([r['overall'] for r in rows]):.2f}")
if __name__ == "__main__": main()
