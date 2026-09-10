"""Cluster errors -> failures.md skeleton. --help/--sample-n/--seed/--out."""
import argparse, pandas as pd
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="results/metrics.json"); ap.add_argument("--judge", default="results/judge_scores.csv")
    ap.add_argument("--pred", default="results/predictions_main.jsonl"); ap.add_argument("--golden", default="data/golden.csv")
    ap.add_argument("--out", default="results/failures.md"); ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    g = pd.read_csv(a.golden, dtype={"id": str}); p = pd.read_json(a.pred, lines=True, dtype={"id": str})
    j = pd.read_csv(a.judge, dtype={"id": str})
    m = g.merge(p, on="id").merge(j, on="id", how="left")
    wrong = m[m.intent_gold != m.intent]
    conf = wrong.groupby(["intent_gold","intent"]).size().sort_values(ascending=False).head(5)
    low = m[m.overall <= 2].head(5) if "overall" in m.columns else m.head(0)
    hal = m[m.hallucinated == "yes"].head(5) if "hallucinated" in m.columns else m.head(0)
    L = ["# Failure mining (author: pick top-5 with real examples)", ""]
    L.append("## Top intent confusions"); L.append(conf.to_string() if len(conf) else "(none)")
    L.append("\n## Low judge scores (overall<=2)"); 
    for _, r in low.iterrows(): L.append(f"- id={r.id} gold={r.intent_gold} pred={r.intent} cust={str(r.customer_text)[:140]} reply={str(r.reply)[:140]}")
    L.append("\n## Hallucinated=yes");
    for _, r in hal.iterrows(): L.append(f"- id={r.id} reply={str(r.reply)[:160]}")
    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"wrote {a.out}")
if __name__ == "__main__": main()
