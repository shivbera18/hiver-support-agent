# Customer Support Agent — AppleSupport

An AI support agent for AppleSupport Twitter conversations. It classifies each incoming customer
message into a small intent set defined from the data, drafts a reply grounded in how the brand
historically resolved similar issues, and decides auto-handle vs. escalate with a stated reason.
Built on the Kaggle Customer Support on Twitter data (thoughtvector/customer-support-on-twitter,
200k-row subsample, seed 42). Banking77 wording analogues informed the intent prompt; no training
on Banking77.

## 1. Reproduce headline results (< 15 min, CPU-only laptop)

```bash
make setup && make data && make eval
```

What each step does and costs (measured on a Windows 11 laptop, Python 3.11):

| target | command chain | time |
|---|---|---|
| `make setup` | `pip install -r requirements.txt` | ~4 min first time |
| `make data` | `download_data.py` (kagglehub, 200k-row sample, seed 42) -> `eda.py` -> `build_threads.py --brand AppleSupport` | ~2 min after download |
| `make eval` | baselines -> TF-IDF agent -> `run_eval` x3 -> heuristic judge x3 -> `agreement` -> `failure_mine` | ~62 s, offline, no keys |
| `make eval-llm` | fusion agent (gpt-5-mini) -> `run_eval` -> LLM judge x4 systems -> `agreement` | ~10 min agent + ~35 min/judged system, ~350k tokens agent run, needs keys |
| `make report-check` | golden 150-250 rows, artifacts exist, report <= 3000 words | instant |

With keys (`OPENAI_API_KEY`, optional `GEMINI_API_KEY` + `GEMINI_ENABLE` in `.env`, see `.env.example`
-- secrets stay out of git): `classify`/`draft_reply` use **gpt-5-mini** primary with a Gemini cascade
(asked only when GPT is silent or conf < 0.7) and TF-IDF last resort. Without keys the pipeline runs
the documented fallback path (TF-IDF classify + retrieval replies + heuristic judge) and prints
`llm=fallback`. Every LLM output is pydantic-validated (`IntentOut`/`JudgeOut`); every call is logged
to `results/llm_usage.json` with an abort gate (`LLM_MIN_SUCCESS=0.8`) that refuses to label
fallback output as LLM results. Full 200-row fusion run: ~350k tokens, 398/400 calls OK.

Manual download fallback: get `twcs.csv` from the Kaggle dataset page, save as `data/twcs_raw.csv`,
then run `make data`. Expected raw header:
`tweet_id,author_id,created_at,text,inbound,response_tweet_id,in_response_to_tweet_id`
(if it differs, adapt the field map in `scripts/build_threads.py` only).

## 2. Repo layout

```
scripts/          download_data, eda, build_threads, make_golden,
                  baseline_trivial, baseline_tfidf, agent, run_eval, judge, agreement, failure_mine
prompts/          intent_prompt, reply_prompt, judge_rubric, escalate_policy (policy also coded in agent.py)
data/             twcs_raw.csv (200k sample), threads_sample.csv (21,602 AppleSupport threads),
                  golden.csv (200 hand-labelled), golden_guide.md (sampling + labelling note)
results/          predictions_{baseline_trivial,baseline_tfidf,main,fusion}.jsonl,
                  metrics{,_trivial,_tfidf,_fusion}.json, judge_{trivial,tfidf,main,fusion}.csv,
                  human_scores.csv (50 blind rows), agreement{,_fusion}.json, failures.md,
                  eda_summary.csv, brand_volume.png
report/           report.md (<=6 pages), decision_log.md (15 decisions)
EVAL.md           harness definition: metrics, splits, commands, fallback status
```

Conventions: Python 3.10+, `seed=42` everywhere, paths relative to repo root, every script supports
`--help`, `--sample-n`, `--seed`, `--out`. No notebooks, no service code -- this is a batch
evaluation pipeline by design (see "What was NOT built").

## 3. Brand pick and intent taxonomy

EDA (`results/eda_summary.csv`, `results/brand_volume.png`): top outbound brands are AmazonHelp
(12,188) and AppleSupport (7,555 per 200k sample); median customer message 109 chars; ~0.8%
non-English proxy; ~51% orphan inbound (no parent context -- the agent must work single-turn).
**Pick: AppleSupport** -- largest English troubleshooting traffic, scopable intents, minimal
moderation risk vs. airlines. Fallback rule (not triggered): AmazonHelp if AppleSupport < 8000 inbound.

| intent | covers |
|---|---|
| `setup_howto` | initial setup, pairing, how-to-use |
| `battery_performance` | battery drain, slow, overheating |
| `software_update` | iOS/macOS update failure/stuck/install |
| `account_icloud` | Apple ID, login, iCloud sync/backup, password |
| `hardware_damage_repair` | cracked screen, water damage, repair status |
| `warranty_applecare` | coverage, AppleCare, replacement eligibility |
| `app_store_itunes` | App Store download, purchase, subscription |
| `connectivity` | Wi-Fi, Bluetooth, cellular, AirDrop |
| `billing_refund` | charge, refund, receipt, payment |
| `other_offtopic` | praise, spam, non-support, incomprehensible, multi-issue ties |

Tie-breaks (enforced by all classifiers, see `data/golden_guide.md`): multi-intent -> the one
requiring action; sarcasm/praise + complaint -> complaint; non-English -> `other_offtopic` + escalate.

## 4. System and baselines

- `classify(text, context) -> (intent, conf)` -- gpt-5-mini JSON mode (pydantic `IntentOut`) with a
  Gemini cascade (only when GPT is silent or conf < 0.7), else TF-IDF + LogReg with golden
  tie-breaks baked in (short/non-English/URL-only -> other; charge-words -> billing).
- `draft_reply(text, context, intent, retrieved) -> str` -- gpt-5-mini (<=280 chars, references
  retrieved history, ends with one next step, never promises refunds or asks for passwords), else
  top-1 retrieved brand reply. PII regex post-filter (-> `[REDACTED]` + force escalate).
- `decide(text, intent, conf) -> (escalate, reason)` -- coded policy: billing intent always
  escalates; risk stems (charg/payment/scam/phish/expir/...) escalate; `conf < 0.55` escalates.
  Threshold tuned once on the B1 dev split (recall >= 0.85 on the risk subset), never re-tuned.

Retrieval: TF-IDF cosine over training-split brand replies augmented with outbound brand tweets
(the random subsample breaks most reply links -- only ~2% of inbound rows carry a linked reply).
Time-ordered 80/20 split, no shuffle (no temporal leakage). Golden IDs asserted out of every
train/retrieval pool.

Baselines (built first, so the agent has a bar): **B0 trivial** -- always `other_offtopic`, fixed
canned reply, always escalate. **B1 simple** -- TF-IDF + LogReg on weak keyword labels, nearest-
neighbour reply (min cosine 0.15 else canned), escalate if `max_proba < 0.60` or risk regex.

| system | intent-acc | macro-F1 | esc-P / esc-R | ROUGE-L | judge-overall |
|---|---|---|---|---|---|
| B0 trivial | 0.415 | 0.059 | 0.160 / 1.000 | 0.407 | 3.47 (gpt-5-mini judge) |
| B1 TF-IDF + LogReg | 0.660 | 0.572 | 0.177 / 0.563 | 0.196 | 2.23 (gpt-5-mini judge) |
| main TF-IDF + intent-filtered retrieval | 0.705 | 0.661 | 0.238 / 0.625 | 0.119 | 1.86 (gpt-5-mini judge) |
| **fusion (gpt-5-mini + cascade)** | **0.805** | **0.778** | **0.316 / 0.563** | **0.139** | **2.69 (gpt-5-mini judge)** |

Fusion: +10pp intent accuracy over the TF-IDF main (Wilson 95% CI 0.74-0.86). Weakest fusion
intents: setup_howto F1 0.50 (n=6), billing_refund 0.55. Checks: fusion > main > B1 > B0 on intent
accuracy; risk-subset escalation recall 14/15 = 0.93 (TF-IDF main). Ablations: threshold 0.40 ->
esc-P/R 0.27/0.53, 0.70 -> 0.18/0.69 (0.55 is the sane middle); no-retrieval collapses judge 2.46 ->
1.82 while ROUGE-L jumps 0.119 -> 0.407 -- retrieval is what makes replies useful, and lexical
metrics punish it for not copying reference templates.

## 5. Golden set (200 hand-labelled examples + sampling note)

`data/golden.csv`: 200 rows -- 15/intent weak-label buckets x 10 (= 150) + 30 uniform random + 20
adversarial (short, ALL-CAPS, sarcasm, non-English, multi-intent). Columns:
`id,brand,customer_text,context_parent_text,intent_gold,reply_reference,escalate_gold,escalate_reason_gold,annotator,notes`.
All 200 hand-labelled by the author against `data/golden_guide.md` (~60 weak labels corrected);
a 50-row second pass flipped 1 label (charge tie-break) and gives intra-annotator intent kappa 0.97
(same-session -- reported as an upper bound). Sampling/labelling note lives at the bottom of
`data/golden_guide.md`. Final distribution: other_offtopic 83, software_update 24, connectivity 17,
account_icloud 16, app_store_itunes 12, hardware_damage_repair 11, billing_refund 11,
battery_performance 10, warranty_applecare 10, setup_howto 6; escalate rate 0.16.

## 6. Evaluation harness + LLM judge + human agreement

`scripts/run_eval.py`: intent accuracy, macro-F1, per-intent F1; escalation P/R/F1 + confusion;
ROUGE-L + TF-IDF cosine (+ BERTScore hook -- null on this box, torch DLL unloadable, logged not
silent); escalation rate, mean reply length. Deterministic except the LLM steps.
`scripts/judge.py`: blinded rubric (`prompts/judge_rubric.txt` -- groundedness, helpfulness,
tone-brand-fit, safety, overall = weakest dim, hallucinated flag), pydantic-validated
(`conint(1,5)`), order Gemini -> GPT -> heuristic with retries, per-row model tags.
`scripts/agreement.py`: the author blind-scored a random 50-row subset
(`results/human_scores.csv`) before seeing judge output. Outcome
(`results/agreement_fusion.json`): judge-vs-human Spearman rho = -0.15, kappa = -0.26 on fusion
rows -- below the 0.50 bar, so every judge number in this repo stays **directional-only**, even
with a real LLM judge. This file is the required agreement evidence; EVAL.md records the status.

## 7. Report: framing, failure modes, misleading number, next week

Top-5 (one real example + hypothesis each, full detail in `report/report.md`): (1) off-topic "how
do I" queries -> money/setup intents (7x, persists under fusion) -- needs a brand-relevance gate;
(2) resolved/thank-you notes -> hardware (5x) -- needs a resolved-statement detector; (3)
functional failures read as broken hardware (4x) -- ambiguous even for humans; (4) native tweet
truncation (~5 rows labelled on fragments) -- irreducible without rehydration; (5) fusion still
misses setup_howto (F1 0.50, n=6) and billing_refund (0.55) -- needs intent-filtered retrieval
plus per-intent thresholds.

Mandatory honesty section: ROUGE-L prefers the canned baseline (reference-template artifact);
n = 200 means ~6pp confidence width; the LLM judge disagrees with the human (rho = -0.15) so all
judge numbers stay directional-only; 2017 tweets vs current policies; single-brand/single-turn
scope. Next week: 100 more adversarial labels, per-intent thresholds, brand-relevance gate,
human-in-the-loop queue mock measuring time-to-resolution. `report/decision_log.md` lists the 15
non-obvious decisions and what was rejected for each.

## 8. What was not built

No live posting or streaming, no multi-brand support, no fine-tuning, no multilingual handling,
no API server/DB/frontend. Banking77: wording analogues in the intent prompt only, never trained on.
SBERT embeddings and BERTScore: attempted, unloadable on this box (torch DLL) -- TF-IDF retrieval
## 9. Borrowings and references

Borrowed and cited: Kaggle Customer Support on Twitter (thoughtvector), Banking77
(Casanueva et al. 2020, wording analogues only), scikit-learn (TF-IDF/LogReg), rouge-score,
OpenAI gpt-5-mini, Google Gemini (flash-latest, quota-limited). AI coding assistance used
throughout; every line was read, repaired, and verified by the author (see commit history).
