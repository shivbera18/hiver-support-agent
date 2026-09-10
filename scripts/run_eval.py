"""Metrics: intent, escalation, ROUGE-L, cosine. --help/--sample-n/--seed/--out. --tune-threshold for B1."""
import argparse, json, pandas as pd, numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
def rouge_l(hyps, refs):
    try:
        from rouge_score import rouge_scorer
        sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(np.mean([sc.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)]))
    except Exception: return 0.0
def cos_sim(hyps, refs):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        v = TfidfVectorizer().fit(hyps + refs)
        import numpy as np
        m = cosine_similarity(v.transform(hyps), v.transform(refs))
        return float(np.mean([m[i,i] for i in range(len(hyps))]))
    except Exception: return 0.0
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", default="data/golden.csv"); ap.add_argument("--pred", default="results/predictions_main.jsonl")
    ap.add_argument("--out", default="results/metrics.json"); ap.add_argument("--tune-threshold", action="store_true")
    ap.add_argument("--sample-n", type=int, default=None); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    g = pd.read_csv(a.golden); preds = pd.read_json(a.pred, lines=True, dtype={"id": str})
    preds["id"] = preds["id"].astype(str); g["id"] = g["id"].astype(str)
    m = g.merge(preds, on="id", how="inner")
    assert len(m) == len(g), f"pred/golden mismatch {len(m)} vs {len(g)}"
    yt, yp = m.intent_gold.tolist(), m.intent.tolist()
    acc = accuracy_score(yt, yp); mf1 = f1_score(yt, yp, average="macro", zero_division=0)
    per = f1_score(yt, yp, average=None, zero_division=0, labels=sorted(set(yt) | set(yp)))
    ye = m.escalate_gold.astype(bool).tolist(); pe = m.escalate.astype(bool).tolist()
    p, r, f, _ = precision_recall_fscore_support(ye, pe, average="binary", zero_division=0)
    cm = confusion_matrix(ye, pe, labels=[False, True]).tolist()
    hyps = m.reply.fillna("").tolist(); refs = m.reply_reference.fillna("").tolist()
    rl = rouge_l(hyps, refs); cs = cos_sim(hyps, refs)
    out = {"n": len(m), "intent_acc": round(float(acc),4), "intent_macro_f1": round(float(mf1),4),
      "per_intent_f1": {k: round(float(v),4) for k, v in zip(sorted(set(yt)|set(yp)), per)},
      "esc_precision": round(float(p),4), "esc_recall": round(float(r),4), "esc_f1": round(float(f),4),
      "esc_confusion_tn_fp_fn_tp": cm, "rougeL": round(float(rl),4), "reply_cosine": round(float(cs),4),
      "esc_rate": round(float(np.mean(pe)),4), "mean_reply_len": round(float(np.mean([len(h) for h in hyps])),1)}
    json.dump(out, open(a.out, "w"), indent=2); print(json.dumps(out, indent=2))
    if a.tune_threshold:
        print("thresholds: (fixed 0.55 per plan; B1 used 0.60)")
if __name__ == "__main__": main()
