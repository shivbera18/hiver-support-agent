# Harness
Split: time-ordered 80/20 (no shuffle) for weak-label training; golden 200 rows held out, leakage-asserted.
Metrics: intent accuracy, macro-F1, per-intent F1 (sklearn); escalation P/R/F1 + confusion; ROUGE-L (rouge-score) + TF-IDF cosine + BERTScore hook (null on this box -- torch DLL unloadable, logged not silent); esc rate, mean reply length.
Classify/reply: gpt-5-mini primary (pydantic IntentOut), gemini-flash-latest cascade (GEMINI_ENABLE, conf<0.7 or silent), TF-IDF last resort. Per-call token ledger (results/llm_usage.json) + abort gate LLM_MIN_SUCCESS=0.8.
Judge: blinded (no system names in prompt; assert on main/baseline), rubric prompts/judge_rubric.txt (groundedness/helpfulness/tone/safety 1-5 bounded, overall = weakest dim, hallucinated flag). Order: gemini -> gpt-5-mini -> heuristic, retries, per-row model tags, mixed-scale abort (80% LLM rows or refuse). Same 200 rows judged per system (results/judge_*_llm.csv).
Human: 50-row blind subset scored before seeing judge output (results/human_scores.csv). Bars: intent re-label kappa>=0.60, judge-vs-human spearman>=0.50 else directional-only. Current: intent kappa 0.97 (same-session upper bound); fusion judge rho=-0.15/kappa=-0.26 -> directional-only holds with a real LLM judge.
Threshold 0.55 tuned once on B1 dev (risk-recall >= 0.85); never re-tuned per system.
Commands: `make eval` (keyless fallback, ~62 s offline) vs `make eval-llm` (full gpt-5-mini fusion + LLM judge, ~350k tokens, needs keys). No keys -> TF-IDF path, rows marked fallback.
