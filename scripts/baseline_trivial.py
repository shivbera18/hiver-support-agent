"""B0: majority intent + canned reply + always escalate. --help/--sample-n/--seed/--out."""
import argparse, json, pandas as pd
CANNED = "Thanks for reaching out \u2014 could you share your device model and iOS version plus what you\u2019ve tried so far?"
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/threads_sample.csv")
    ap.add_argument("--golden", default="data/golden.csv")
    ap.add_argument("--out", default="results/predictions_baseline_trivial.jsonl")
    ap.add_argument("--sample-n", type=int, default=None); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    g = pd.read_csv(a.golden)
    if a.sample_n: g = g.head(a.sample_n)
    with open(a.out, "w", encoding="utf-8") as f:
        for _, r in g.iterrows():
            f.write(json.dumps({"id": str(r["id"]), "intent": "other_offtopic", "reply": CANNED, "escalate": True, "reason": "policy:trivial-always-escalate", "conf": 0.0}) + "\n")
    print(f"wrote {a.out} rows={len(g)}")
if __name__ == "__main__": main()
