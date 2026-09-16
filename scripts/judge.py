"""LLM judge (gemini primary, gpt second) blinded on rubric; heuristic fallback. --help/--sample-n/--seed/--out."""
import argparse, json, os, re, pandas as pd, numpy as np
try:
    from dotenv import load_dotenv; load_dotenv(".env", override=True)
except Exception: pass
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-mini")
try:
    from pydantic import BaseModel, confloat, conint
    from typing import Literal
    Dim = conint(ge=1, le=5)
    class JudgeOut(BaseModel):
        groundedness: Dim; helpfulness: Dim; tone: Dim; safety: Dim; overall: Dim
        hallucinated: Literal["yes", "no"]
    _HAVE_PYDANTIC = True
except Exception: _HAVE_PYDANTIC = False
def heuristic(r):
    cand, cust = str(r.reply), str(r.customer_text)
    STOP = set("the,a,an,to,for,and,or,of,on,in,is,are,you,your,we,us,our,it,its,this,that,with,what,when,how,have,has,had,do,does,did,can,will,please,hi,hey,thanks,thank,my,me,i,so,but,if,then,than,there,their,they,them,at,by,from,up,out,about,into,over,after".split(","))
    cw = {w.strip(".,!?;:'\"()").lower() for w in cust.split()} - STOP
    rw = {w.strip(".,!?;:'\"()").lower() for w in cand.split()} - STOP
    cw = {w for w in cw if len(w) > 2}; rw = {w for w in rw if len(w) > 2}
    overlap = len(cw & rw)
    grounded = 5 if overlap >= 4 else (4 if overlap == 3 else (3 if overlap == 2 else (2 if overlap == 1 else 1)))
    if str(r.get("intent", "")) == str(r.get("intent_gold", "")) and grounded < 5: grounded += 1
    helpful = 5 if ("?" in cand and ("dm" in cand.lower() or "http" in cand or "setting" in cand.lower())) else (4 if "?" in cand else (3 if len(cand) > 60 else 2))
    tone = 2 if re.search(r"\b(damn|stupid|idiot|hate you)\b", cand, re.I) or (cand.isupper() and len(cand) > 10) else 5
    hall = "yes" if re.search(r"order #\d+|guarantee|promise.*refund|will refund", cand, re.I) else "no"
    safety = 2 if (hall == "yes" or re.search(r"password|ssn|card number", cand, re.I)) else 5
    overall = min(grounded, helpful, tone, safety)
    return [grounded, helpful, tone, safety, overall, hall]
_JUSE = {"calls": 0, "ok": 0, "llm_rows": 0}
def _gretry(fn, tries=3):
    import time
    for i in range(tries):
        try: return fn()
        except Exception as e:
            print(f"judge try {i+1} failed:", str(e)[:100]); time.sleep(3 * (i + 1))
    return None

def llm_judge(rubric, r):
    global _JUSE
    msg = "GOLDEN REFERENCE: " + str(r.reply_reference) + "\nCANDIDATE REPLY: " + str(r.reply) + "\nCUSTOMER: " + str(r.customer_text)[:300] + "\nReturn JSON only: {\"groundedness\": 1-5, \"helpfulness\": 1-5, \"tone\": 1-5, \"safety\": 1-5, \"overall\": 1-5, \"hallucinated\": \"yes\"/\"no\"}"
    order = []
    if os.environ.get("GEMINI_API_KEY") and os.environ.get("GEMINI_ENABLE") and _HAVE_PYDANTIC: order.append("gemini")
    if os.environ.get("OPENAI_API_KEY") and _HAVE_PYDANTIC: order.append("gpt")
    for which in order:
        _JUSE["calls"] += 1
        try:
            if which == "gemini":
                import time
                from google import genai
                g = genai.Client()
                resp = _gretry(lambda: g.models.generate_content(model=GEMINI_MODEL, contents=rubric + "\n" + msg, config={"http_options": {"timeout": 90000}}))
                if resp is None: raise RuntimeError("gemini judge unreachable")
                t = re.sub(r"^```(json)?|```$", "", resp.text.strip()).strip()
                d = JudgeOut.model_validate_json(t)
                _JUSE["ok"] += 1; _JUSE["llm_rows"] += 1
                return [d.groundedness, d.helpfulness, d.tone, d.safety, d.overall, d.hallucinated, "gemini-llm"]
            import time
            from openai import OpenAI
            c = OpenAI(max_retries=0)
            resp, err = None, ""
            for i in range(4):
                try:
                    resp = c.chat.completions.create(model=OPENAI_MODEL, response_format={"type": "json_object"},
                        messages=[{"role": "system", "content": rubric}, {"role": "user", "content": msg}],
                        max_completion_tokens=1000, timeout=90); break
                except Exception as e:
                    err = str(e)[:100]; print(f"gpt judge try {i+1}: {err}"); time.sleep(5 * (i + 1))
            if resp is None: raise RuntimeError(err)
            d = JudgeOut.model_validate_json(resp.choices[0].message.content)
            _JUSE["ok"] += 1; _JUSE["llm_rows"] += 1
            return [d.groundedness, d.helpfulness, d.tone, d.safety, d.overall, d.hallucinated, "gpt-llm"]
        except Exception as e:
            print(f"JUDGE {which} FAIL id={r['id']}: {str(e)[:120]}")
    return None
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="results/predictions_main.jsonl"); ap.add_argument("--golden", default="data/golden.csv")
    ap.add_argument("--out", default="results/judge_scores.csv"); ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    g = pd.read_csv(a.golden); preds = pd.read_json(a.pred, lines=True)
    preds["id"]=preds["id"].astype(str); g["id"]=g["id"].astype(str)
    m = g.merge(preds, on="id").sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    if a.sample_n: m = m.head(a.sample_n)
    rubric = open("prompts/judge_rubric.txt").read()
    assert "main" not in rubric.lower() and "baseline" not in rubric.lower(), "judge prompt not blind"
    rows = []
    for _, r in m.iterrows():
        d = llm_judge(rubric, r)
        if d is None:
            h = heuristic(r); d = h + ["heuristic-fallback"]
        rows.append({"id": r["id"], "groundedness": d[0], "helpfulness": d[1], "tone": d[2], "safety": d[3], "overall": d[4], "hallucinated": d[5], "model": d[6]})
    import json as _j
    rate = _JUSE["ok"] / max(_JUSE["calls"], 1)
    print(f"JUDGE SUMMARY calls={_JUSE['calls']} ok={_JUSE['ok']} llm_rows={_JUSE['llm_rows']}/{len(rows)}")
    _j.dump(_JUSE, open("results/judge_usage.json", "w"), indent=2)
    floor = float(os.environ.get("LLM_MIN_SUCCESS", "0.8"))
    if _JUSE["calls"] > 0 and _JUSE["llm_rows"] / max(len(rows), 1) < floor:
        raise SystemExit(f"ABORT: only {_JUSE['llm_rows']}/{len(rows)} rows llm-judged -- refusing mixed-scale scores")
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"wrote {a.out} mean_overall={np.mean([r['overall'] for r in rows]):.2f}")
if __name__ == "__main__": main()
