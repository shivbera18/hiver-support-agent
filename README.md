# Hiver SDE Intern -- AppleSupport AI Agent

Hybrid RAG support agent: classify -> retrieve top-3 brand resolutions -> draft reply -> decide escalate. Brand: **AppleSupport** (7555 outbound replies in 200k sample; fallback rule kept AmazonHelp if <8000 inbound -- not triggered).

## Repro (<15 min, CPU-only)

```bash
make setup && make data && make eval
```

`make data` downloads twcs via kagglehub (or place `twcs.csv` at `data/twcs_raw.csv`, see `data/README_DATA.md`), samples 200k rows (seed 42), builds AppleSupport threads. Without `OPENAI_API_KEY` credits, `make eval` runs the documented fallback path (TF-IDF classify + retrieval replies + heuristic judge) and prints `llm=fallback`. Full run ~3 min on a laptop. `make report-check` gates golden size (150-250), artifacts, and report length.

## Results (n = 200 golden, `results/metrics*.json`)

| system | intent-acc | macro-F1 | esc-P / esc-R | ROUGE-L | judge-overall |
|---|---|---|---|---|---|
| B0 trivial | 0.415 | 0.059 | 0.160 / 1.000 | 0.407 | 1.53 |
| B1 TF-IDF + LogReg | 0.660 | 0.572 | 0.177 / 0.563 | 0.196 | 2.35 |
| main | 0.705 | 0.661 | 0.238 / 0.625 | 0.105 | 2.40 |

Risk-subset escalation recall 14/15 = 0.93. Judge is heuristic (`model=heuristic-fallback`): judge-vs-human rho = 0.25, kappa = 0.30 -- directional-only. Full analysis: `report/report.md`. Decisions: `report/decision_log.md`. Harness spec: `EVAL.md`.

## Borrowings

Kaggle twcs (thoughtvector/customer-support-on-twitter); Banking77 (Casanueva et al. 2020) wording analogues only; scikit-learn TF-IDF/LogReg; rouge-score; MiniLM listed but unused offline (TF-IDF retrieval fallback per EVAL.md).
