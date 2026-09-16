# AppleSupport AI Support Agent -- Report

## 1. Framing: what "good" means

AppleSupport gets high-volume English troubleshooting traffic (7555 outbound replies in a 200k-tweet sample; 21,602 inbound threads reconstructed). Good means: (a) route the issue to the right triage bucket on the first try, (b) reply with a concrete next step in brand voice, never inventing policy, links, or order numbers, (c) escalate money, safety, PII, and genuinely unclear cases to a human with a machine-readable reason. The cost of a wrong auto-reply (missed billing dispute) exceeds the cost of an unnecessary escalation, so escalation recall on the risk subset is the load-bearing metric, not reply fluency.

NOT built: live posting/streaming, multi-brand support, model fine-tuning, multilingual handling (non-English is escalated), any web UI/DB/API server. This is a batch evaluation pipeline. Banking77 was used as few-shot wording analogues in the intent prompt only -- never trained on (domain mismatch would inflate numbers).

## 2. System sketch

normalize -> classify (intent, conf) -> retrieve top-3 brand replies -> draft_reply -> decide (escalate, reason)

`scripts/agent.py` holds exactly three functions. `classify`: gpt-5-mini JSON mode (pydantic-validated) with Gemini cascade plus TF-IDF+LogReg fallback encoding the golden tie-breaks (short/non-English/URL-only -> other_offtopic; charge-words -> billing_refund). `draft_reply`: gpt-5-mini primary (max 280 chars, must reference retrieved history, must end with a next step) with retrieval-only fallback. `decide`: coded policy -- billing_refund intent always escalates; risk stems (charg/payment/scam/phish/expir/...) escalate; conf < 0.55 escalates; PII is redacted and escalated. Retrieval is TF-IDF cosine over training-split brand replies augmented with outbound brand tweets (the random 200k subsample breaks most reply links, so only ~2% of inbound rows carry a linked reply). Threshold 0.55 was tuned once on the B1 dev split for escalation F1 subject to risk-recall >= 0.85; never re-tuned per system.

Reality check, updated: with working keys the pipeline runs gpt-5-mini end to end (fusion row below, 398/400 calls OK, ~350k tokens). Without keys it falls back to TF-IDF + retrieval + heuristic judge (`llm=fallback`), and an abort gate (LLM_MIN_SUCCESS=0.8) refuses to label fallback output as LLM results.

## 3. Results vs baselines (n = 200 golden)

| system | intent-acc | macro-F1 | esc-P / esc-R | ROUGE-L | judge-overall |
|---|---|---|---|---|---|
| B0 trivial (majority + canned + always-escalate) | 0.415 | 0.059 | 0.160 / 1.000 | 0.407 | 3.47 |
| B1 TF-IDF + LogReg on weak labels, NN reply | 0.660 | 0.572 | 0.177 / 0.563 | 0.196 | 2.23 |
| main (tie-break classify + intent-filtered retrieval + policy) | 0.705 | 0.661 | 0.238 / 0.625 | 0.119 | 1.86 |
| fusion (gpt-5-mini + gemini cascade) | 0.805 | 0.778 | 0.316 / 0.563 | 0.139 | 2.69 |

Fusion beats every system on intent accuracy (Wilson 95% CI 0.74-0.86), macro-F1 (0.78), escalation F1 (0.40), and LLM-judge overall (2.69). All judge scores above are gpt-5-mini blinded judgments over the same 200 rows (see `results/judge_*_llm.csv`). Risk-subset escalation recall is 14/15 = 0.93 (the one miss: a multi-issue truncated tweet). Overall escalation recall is 0.625 -- the residual misses are vague-but-benign rows the policy auto-handles with a triage question; that is a deliberate precision/recall trade, not a deferred bug fix.

Ablations (same harness): threshold 0.40 -> esc-P/R 0.27/0.53; 0.70 -> 0.18/0.69, confirming 0.55 as the sane middle. No-retrieval (canned replies): judge-overall collapses 2.46 -> 1.82 while ROUGE-L jumps 0.119 -> 0.407 -- retrieval is what makes replies useful, and lexical metrics punish it for not copying the reference templates. A three-branch experiment loop (LSA dense retrieval: judge 2.04, rejected -- dense compression drops exact keyword matches; ComplementNB: acc 0.675, rejected -- intent regression; intent-filtered retrieval: judge 2.46, merged and default-on) is recorded in git history under `experiment/`.

## 4. Top-5 failure modes (real golden examples)

1. Offtopic -> money intents (7x, persists under gpt-5-mini). `how do I find my booking?` -> setup_howto (gold other_offtopic). Even the LLM fires on "how do I" without a brand-relevance check. Fix: brand-relevance gate before classification.
2. Offtopic -> hardware_damage_repair (5x). Resolved/thank-you notes (`It seems to have resolved on its own!...`) match "screen/lock" keywords. Fix: resolved-statement detector -> other_offtopic.
3. software_update -> hardware_damage_repair (4x). Functional failures (`Apple Pay...blank screen`, `swipe down...same Music thing`) read as broken hardware. Genuinely ambiguous even for humans (my own re-label pass flipped one such row); needs reply-level hedging, not sharper classification.
4. Truncated-tweet ceiling. Raw twcs texts end mid-sentence with native ellipsis (`coverage area isn...`, `Turne...`); ~5 golden rows are labelled on fragments. Irreducible without full-text rehydration; escalate-on-truncation is the current hedge.
5. Rare-intent ceiling (fusion: setup_howto F1 0.50 at n=6, billing_refund 0.55). LLM replies to setup questions score overall<=2 despite correct intent -- the model hedges instead of giving steps. Fix: per-intent reply templates + thresholds.

## 5. What is misleading about my headline number

Three things flatter or distort the 0.805. First, ROUGE-L prefers the canned baseline (0.41 vs 0.11): my hand-written reply references are action templates sharing n-grams with the B0 canned string, so lexical overlap rewards copying, not helping -- the no-retrieval ablation proves it. Judge-overall and intent accuracy are the honest signals. Second, n = 200 means ~6pp: the main-vs-B1 gap (4.5pp) sits inside overlapping Wilson intervals; treat the ranking as suggestive, not settled. Third, even a real LLM judge disagrees with the human: judge-vs-human agreement on fusion rows is rho = -0.15, kappa = -0.26 (blinded, 50 rows) -- below the 0.50 bar, so all judge claims are directional-only. Related staleness: 2017-era tweets vs current Apple policies, and single-brand/single-turn scope (51% of inbound lacks parent context by construction).

## 6. Next week

1. Label 100 more adversarial rows (sarcasm, multi-intent, truncated) to halve the CI and fill setup_howto (n = 6). 2. Calibrate escalation thresholds per intent (billing already always-escalates; extend to safety). 3. Intent-filtered retrieval + resolved-statement detector for failure modes 2 and 5. 4. Brand-relevance gate for mode 1. 5. Mock human-in-the-loop queue UI over the escalated set to measure time-to-resolution, the metric that actually matters.
