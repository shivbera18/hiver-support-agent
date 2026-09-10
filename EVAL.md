# Harness
Split: time-ordered 80/20 (no shuffle) for weak-label training; golden 200 rows held out, leakage-asserted.
Metrics: intent accuracy, macro-F1, per-intent F1 (sklearn); escalation P/R/F1 + confusion; ROUGE-L (rouge-score) + embedding-cosine fallback if bert-score download fails; esc rate, mean reply len, redactions.
Judge: gpt-4o-mini temp 0, blinded, 4 dims 1-5 + overall + hallucinated flag. Human scores 50-item subset before seeing judge.
Bars: intent re-label kappa>=0.60, judge-vs-human spearman>=0.50 else judge is directional-only. Threshold 0.55 tuned once on B1 dev for esc-F1 with recall>=0.85 on risk subset; not re-tuned per system.
Commands: see Makefile `make eval`. No-API fallback: TF-IDF path, rows marked model=fallback.
Fallback status (2026-09-10): no LLM credits, so classify/draft run TF-IDF+retrieval and judge runs heuristics (customer-overlap grounding + intent-match, overall = weakest dim, model=heuristic-fallback). Agreement: intent kappa 0.97 (same-session upper bound), judge rho 0.08/kappa 0.07 on final outputs -> judge directional-only. Intent-filtered retrieval adopted after experiment loop.
