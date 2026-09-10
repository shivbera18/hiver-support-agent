# Hiver SDE Intern -- AI Support Agent for AppleSupport

An AI customer-support agent built on real Twitter support conversations. Given an incoming customer
message, it (1) classifies it into a self-defined intent set, (2) drafts a reply grounded in how
AppleSupport historically resolved similar issues, and (3) decides auto-handle vs. escalate with a
stated reason -- then proves it can be trusted with a golden set, baselines, and a judged eval harness.

Assignment: `Hiver SDE Intern Assignment.pdf` (in repo root). Source: Kaggle
`thoughtvector/customer-support-on-twitter` (~3M tweets, multi-turn, noisy). Secondary data
(PolyAI/banking77) used for intent wording analogues only -- never trained on.

## 1. Reproduce headline results (< 15 min, CPU-only laptop)

```bash
make setup && make data && make eval
```

What each step does and costs (measured on a Windows 11 laptop, Python 3.11):

| target | command chain | time |
|---|---|---|
| `make setup` | `pip install -r requirements.txt` | ~4 min first time |
| `make data` | `download_data.py` (kagglehub, 200k-row sample, seed 42) -> `eda.py` -> `build_threads.py --brand AppleSupport` | ~2 min after download |
| `make eval` | baselines -> agent -> `run_eval` x3 -> `judge` x3 -> `agreement` -> `failure_mine` | ~62 s |
| `make report-check` | golden 150-250 rows, artifacts exist, report <= 3000 words | instant |

No `OPENAI_API_KEY` credits? The pipeline still runs end to end on the documented fallback path
(TF-IDF classify + retrieval replies + heuristic judge) and prints `llm=fallback`. All numbers in
this README are from that fallback path -- the provisioned key returned 429 (no credits), stated
openly in `report/report.md` section 2. If you add credits, `classify`/`draft_reply` switch to
`gpt-4o-mini` automatically; expect intent accuracy and judge scores to rise.

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
results/          predictions_*.jsonl, metrics*.json, judge_*.csv, human_scores.csv (50 blind rows),
                  agreement.json, failures.md, eda_summary.csv, brand_volume.png
report/           report.md (<=6 pages), decision_log.md (14 decisions)
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

`scripts/agent.py` -- exactly three functions, no framework:

- `classify(text, context) -> (intent, conf)` -- LLM JSON mode, else TF-IDF + LogReg with golden
  tie-breaks baked in (short/non-English/URL-only -> other; charge-words -> billing).
- `draft_reply(text, context, intent, retrieved) -> str` -- LLM (<=280 chars, references retrieved
  history, ends with one next step, never promises refunds or asks for passwords), else top-1
  retrieved brand reply. PII regex post-filter (-> `[REDACTED]` + force escalate).
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

## 5. Results (n = 200 golden; `results/metrics*.json` -- report table matches byte-for-byte)

| system | intent-acc | macro-F1 | esc-P / esc-R | ROUGE-L | judge-overall |
|---|---|---|---|---|---|
| B0 trivial | 0.415 | 0.059 | 0.160 / 1.000 | 0.407 | 1.53 |
| B1 TF-IDF + LogReg | 0.660 | 0.572 | 0.177 / 0.563 | 0.196 | 2.35 |
| main | 0.705 | 0.661 | 0.238 / 0.625 | 0.105 | 2.40 |

Checks: main > B1 > B0 on intent accuracy (Wilson 95% CI 0.64-0.76); risk-subset escalation recall
14/15 = 0.93. Ablations: threshold 0.40 -> esc-P/R 0.27/0.53, 0.70 -> 0.18/0.69 (0.55 is the sane
middle); no-retrieval collapses judge 2.40 -> 1.82 while ROUGE-L jumps 0.105 -> 0.407 -- retrieval
is what makes replies useful, and lexical metrics punish it for not copying reference templates.

## 6. Golden set (deliverable 2)

`data/golden.csv`: 200 rows -- 15/intent weak-label buckets x 10 (= 150) + 30 uniform random + 20
adversarial (short, ALL-CAPS, sarcasm, non-English, multi-intent). Columns:
`id,brand,customer_text,context_parent_text,intent_gold,reply_reference,escalate_gold,escalate_reason_gold,annotator,notes`.
All 200 hand-labelled by the author against `data/golden_guide.md` (~60 weak labels corrected);
a 50-row second pass flipped 1 label (charge tie-break) and gives intra-annotator intent kappa 0.97
(same-session -- reported as an upper bound). Sampling/labelling note lives at the bottom of
`data/golden_guide.md`. Final distribution: other_offtopic 83, software_update 24, connectivity 17,
account_icloud 16, app_store_itunes 12, hardware_damage_repair 11, billing_refund 11,
battery_performance 10, warranty_applecare 10, setup_howto 6; escalate rate 0.16.

## 7. Eval harness + judge + agreement (deliverable 3)

`scripts/run_eval.py`: intent accuracy, macro-F1, per-intent F1; escalation P/R/F1 + confusion;
ROUGE-L + TF-IDF cosine fallback; escalation rate, mean reply length. Deterministic, no API needed.
`scripts/judge.py`: blinded rubric (`prompts/judge_rubric.txt` -- groundedness, helpfulness,
tone-brand-fit, safety, overall = weakest dim, hallucinated flag). `scripts/agreement.py`: the
author blind-scored a random 50-row subset (`results/human_scores.csv`) before seeing judge output.
Outcome (`results/agreement.json`): judge-vs-human Spearman rho = 0.25, kappa = 0.30 -- below the
0.50 bar, so every judge number in this repo is labelled **directional-only**. This file is the
required agreement evidence; EVAL.md records the fallback status.

## 8. Failure modes, misleading number, next week (report sections 4-6)

Top-5 (one real example + hypothesis each, full detail in `report/report.md`): (1) off-topic "how
do I" queries -> money/setup intents (7x) -- needs a brand-relevance gate; (2) resolved/thank-you
notes -> hardware (5x) -- needs a resolved-statement detector; (3) functional failures read as
broken hardware (4x) -- ambiguous even for humans; (4) native tweet truncation (~5 rows labelled on
fragments) -- irreducible without rehydration; (5) mismatched retrieval on rare intents
(setup_howto n = 6) -- needs intent-filtered retrieval.

Mandatory honesty section: ROUGE-L prefers the canned baseline (reference-template artifact);
n = 200 means ~6pp confidence width, so the main-vs-B1 gap is suggestive not settled; the judge is
a heuristic, not an LLM; 2017 tweets vs current policies; single-brand/single-turn scope. Next
week: 100 more adversarial labels, per-intent thresholds, intent-filtered retrieval, relevance gate,
human-in-the-loop queue mock measuring time-to-resolution. `report/decision_log.md` lists the 14
non-obvious decisions and what was rejected for each.

## 9. What was NOT built

No live posting or streaming, no multi-brand support, no fine-tuning, no multilingual handling,
no API server/DB/frontend. Banking77: wording analogues in the intent prompt only, never trained on.

## 10. Borrowings and submission

Borrowed and cited: Kaggle twcs (thoughtvector), Banking77 (Casanueva et al. 2020), scikit-learn,
rouge-score. AI coding assistance used throughout; every line was read, repaired, and verified by
the author (see commit history). Submit via the Notion form with this repo link; no email.
