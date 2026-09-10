"""Judge-vs-human spearman + kappa; intent intra-annotator kappa. --help/--sample-n/--seed/--out."""
import argparse, json, pandas as pd, numpy as np
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="results/judge_scores.csv"); ap.add_argument("--human", default="results/human_scores.csv")
    ap.add_argument("--out", default="results/agreement.json"); ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    j = pd.read_csv(a.judge, dtype={"id": str})
    try: h = pd.read_csv(a.human, dtype={"id": str})
    except FileNotFoundError:
        print(f"no {a.human}; writing template from judge ids (score 50 rows 1-5 BEFORE running again)")
        j.head(50)[["id"]].assign(groundedness="", helpfulness="", tone="", safety="", overall="", hallucinated="").to_csv(a.human, index=False)
        json.dump({"status": "need-human-scores", "n_judge": len(j)}, open(a.out, "w"), indent=2); return
    m = j.merge(h, on="id", suffixes=("_judge", "_human"))
    from scipy.stats import spearmanr
    rho = float(spearmanr(m.overall_judge, m.overall_human).statistic) if len(m) > 2 else 0.0
    jb = (m.overall_judge >= 4).astype(int); hb = (m.overall_human >= 4).astype(int)
    from sklearn.metrics import cohen_kappa_score
    kappa = float(cohen_kappa_score(jb, hb)) if len(set(jb)) > 1 and len(set(hb)) > 1 else 0.0
    exact = float((m.overall_judge == m.overall_human).mean())
    out = {"n_overlap": len(m), "judge_spearman": round(rho,3), "judge_kappa": round(kappa,3),
      "exact_agreement": round(float(exact),3), "intent_kappa": "fill-after-relabel",
      "verdict": "judge-ok" if rho >= 0.50 else "directional-only"}
    try:
        prev = json.load(open(a.out))
        for k in ("intent_kappa", "intent_kappa_note", "n_relabel"):
            if k in prev: out[k] = prev[k]
    except Exception: pass
    json.dump(out, open(a.out, "w"), indent=2); print(json.dumps(out, indent=2))
if __name__ == "__main__": main()
