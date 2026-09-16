# Session Log — Hiver AI Support Agent (AppleSupport)

Covers the full build: greenfield repo → baselines → main agent → golden set → eval harness →
report → git history → experiment loop → LLM-fusion branch (blocked on network outage at the end).
**No secrets in this file** — key material is referenced by length/status only.

## 1. Starting state

- Empty working tree `Hiver/` containing only `Hiver SDE Intern Assignment.pdf` (2 pages, verified).
- Assignment: pick ONE brand from `thoughtvector/customer-support-on-twitter` (~3M tweets), build an
  agent that (1) classifies into a self-defined intent set, (2) drafts replies grounded in brand
  history, (3) decides auto-handle vs escalate with reason — then prove trustworthiness.
- Deliverables: runnable repo (<15 min repro), 150–250 hand-labelled golden rows, eval harness +
  LLM-judge rubric + judge-vs-human agreement, ≤6-page report (≥2 baselines, top-5 failures,
  mandatory "misleading headline" section, next-week plan), 10–15 bullet decision log, Notion-form
  submission. AI assistants allowed; subsample expected.
- Environment: Windows 11, Python 3.11.9, no pandas/sklearn at start. `OPENAI_API_KEY` env var was
  set (later found to have no credits — 429 on every call).

## 2. Repo scaffold (Step 0)

Created exact plan tree, nothing more (no `utils.py`, no `config.yaml`, no Docker/CI/server):

- `README.md`, `requirements.txt` (pinned: `scikit-learn==1.3.2`, `pandas==2.2.2`,
  `numpy==1.26.4`, `sentence-transformers==2.7.0`, `openai>=1.0`, `rouge-score`, `bert-score`,
  `krippendorff`, `pyyaml`, `tqdm`, `matplotlib` + later `pydantic`, `python-dotenv`, `google-genai`),
  `Makefile` (`setup`, `data`, `eval`, `report-check`), `EVAL.md`.
- `scripts/`: `download_data`, `eda`, `build_threads`, `make_golden`, `baseline_trivial`,
  `baseline_tfidf`, `agent`, `run_eval`, `judge`, `agreement`, `failure_mine`.
- `prompts/`: `intent_prompt`, `reply_prompt`, `judge_rubric`, `escalate_policy`.
- `data/`, `results/`, `report/` (report + decision log).
- Conventions: Python 3.10+, `seed=42`, repo-relative paths, every script has
  `--help/--sample-n/--seed/--out`. Scripts only, no notebooks.

## 3. Data: download, EDA, brand pick, threads (Step 1)

- `download_data.py` via `kagglehub`: found the real CSV nested at
  `versions/10/twcs/twcs.csv` (516 MB; the top-level `sample.csv` is a decoy). Patched selector to
  recurse `**/*.csv` and take the largest. Sampled 200,000 rows, seed 42.
- Real header confirmed: `tweet_id,author_id,created_at,text,inbound,response_tweet_id,in_response_to_tweet_id`.
- Two bugs fixed along the way:
  - `eda.py` counted **inbound author_ids** (numeric customer IDs) instead of **outbound brand
    names**. Fixed to count outbound `author_id`.
  - `build_threads.py` had an inverted inbound mask (`is_in = ~is_in` line) → zero brand rows.
    Fixed; then fixed reply linkage: replies resolve via inbound `response_tweet_id`
    (comma-separated, float-cast) with fallback to outbound `in_response_to_tweet_id`.
- EDA (200k sample): 109,275 inbound / 90,725 outbound. Top outbound: AmazonHelp 12,188,
  **AppleSupport 7,555**, Uber_Support 3,957. Median customer length 109 chars, non-ASCII proxy
  0.008, orphan rate 0.514 (~51% single-turn — agent must work without context).
- **Brand: AppleSupport** (pre-decided; AmazonHelp fallback at <8000 inbound not triggered).
- `threads_sample.csv`: 21,602 rows, schema
  `msg_id,brand,customer_text,parent_brand_text,parent_customer_text,created_at,brand_reply_text`.
  Only ~2% carry a linked reply (random subsample breaks thread links) → reply pools were
  augmented with outbound brand tweets (total pool ~7.9k–27.5k).
- Normalization: strip `@Brand`, URLs → `<URL>`, keep emoji, truncate 500 chars + `<TRUNC>`,
  dedup max 3 copies, 30k cap.

## 4. Intent taxonomy (Step 2) — frozen, 10 intents

`setup_howto`, `battery_performance`, `software_update`, `account_icloud`,
`hardware_damage_repair`, `warranty_applecare`, `app_store_itunes`, `connectivity`,
`billing_refund`, `other_offtopic`.
Tie-breaks (in guide + prompt + code): multi-intent → action-requiring intent; sarcasm/praise +
complaint → complaint; non-English / URL-only / too-short → `other_offtopic` + escalate.
Banking77 used as few-shot wording analogues in the intent prompt only — never trained on.

## 5. Baselines (Step 3) — built first

- **B0 trivial** (`baseline_trivial.py`): always `other_offtopic`, fixed canned reply, always
  escalate (`policy:trivial-always-escalate`).
- **B1 simple** (`baseline_tfidf.py`): `TfidfVectorizer(20k, 1–2 grams)` + `LogisticRegression`
  on weak keyword labels, 20k rows, **time-ordered 80/20**; NN reply (cos ≥ 0.15 else canned);
  escalate if `max_proba < 0.60` or risk regex. Threshold 0.55 for main tuned once on B1 dev
  (esc-F1, risk-recall ≥ 0.85); never re-tuned per system.

## 6. Main agent (Step 4) — `scripts/agent.py`, 3 functions, no framework

- `classify(text, context)`: LLM JSON mode (temp 0) → TF-IDF+LogReg fallback with golden
  tie-breaks baked in (short/non-English/URL-only → other @ conf 0.0; charge-words →
  `billing_refund` @ ≥0.65).
- `draft_reply`: LLM (≤280 chars, must reference retrieved history, end with one next step, never
  promise refunds / ask passwords) → top-1 retrieved reply → canned. PII regex post-filter →
  `[REDACTED]` + force escalate. Refund-promise guard → escalate.
- `decide`: coded policy — `billing_refund` intent always escalates; widened risk stems
  (`charg/payment/scam/phish/expir/…`); `conf < 0.55`; PII/non-English/too-short rules.
  Risk recall went 0.80 → 0.93 after widening stems (deliberately not by lowering threshold).
- Retrieval: TF-IDF cosine top-3 over train pool; later **intent-filtered** (per-intent pools,
  shared weak-label map — see §10).

## 7. Golden set (Step 5) — 200 rows, hand-labelled

`make_golden.py`: 15/intent weak buckets × 10 (=150) + 30 random + 20 adversarial = 200.
**All 200 hand-reviewed** (~60 weak labels corrected — weak buckets were noisy, esp. off-topic
leakage into every intent). Notable fixes: earpods→hardware, charge rows→billing per tie-break,
truncated fragments kept with `truncated-ambiguous` notes. Reply references are author-written
action templates per intent (a `{step}` placeholder leak in 6 setup rows was caught and fixed).
One pass-2 re-label flip (R7 app_store→billing). Final: other 83, software_update 24,
connectivity 17, account_icloud 16, app_store 12, hardware 11, billing 11, battery 10,
warranty 10, setup 6; escalate rate 0.16. Sampling/labelling note in `data/golden_guide.md`;
leakage assertion (golden IDs excluded from train/retrieval).

## 8. Eval harness + judge + agreement (Step 6)

- `run_eval.py`: intent acc / macro-F1 / per-intent F1, esc P/R/F1 + confusion, ROUGE-L +
  TF-IDF-cosine (+ BERTScore hook added later), esc rate, reply length. Deterministic, no API.
- `judge.py`: blinded rubric (groundedness/helpfulness/tone/safety, **overall = weakest dim**,
  hallucinated flag). No-credit reality → transparent heuristic fallback
  (`model=heuristic-fallback`: customer-overlap grounding + intent-match bonus), never
  constant-3s. Judge-ablation loop v1→v3 landed on overall=min(dims).
- `agreement.py`: author blind-scored random 50-row subset **before** seeing judge output
  (`human_scores.csv`). Outcome: judge-vs-human **ρ = 0.25 → 0.08** (final outputs), κ ≈ 0.07 —
  below the 0.50 bar → all judge numbers labelled **directional-only** (kept, downgraded, never
  re-sampled). Intent re-label κ = 0.97 (same-session — reported as upper bound).
- `failure_mine.py` → `results/failures.md` skeleton; top-5 written into the report with real
  examples (offtopic→money/setup keywords, resolved-notes→hardware, functional-failure ambiguity,
  native tweet truncation, rare-intent retrieval mismatch).

## 9. Results (n = 200; `results/metrics*.json`, report matches byte-for-byte)

| system | intent-acc | macro-F1 | esc-P / esc-R | ROUGE-L | judge |
|---|---|---|---|---|---|
| B0 trivial | 0.415 | 0.059 | 0.160 / 1.000 | 0.407 | 1.53 |
| B1 TF-IDF | 0.660 | 0.572 | 0.177 / 0.563 | 0.196 | 2.35 |
| main (adopted) | **0.705** (Wilson 0.64–0.76) | **0.661** | **0.238 / 0.625** | 0.119 | **2.46** |

Risk-subset esc recall 14/15 = 0.93. Ablations: thr 0.40 → 0.27/0.53, 0.70 → 0.18/0.69;
no-retrieval judge 2.46 → 1.82 while ROUGE jumps 0.119 → 0.407 (lexical metrics reward copying).
Weakest intents: setup_howto F1 0.33 (n=6), hardware 0.44, billing 0.50.

## 10. Report, decision log, README (Steps 7–8)

- `report/report.md` (939 words, ≤3000): framing + NOT-built, system sketch, results + ablations,
  top-5 failures, **"What is misleading about my headline number"** (ROUGE artifact, ±6pp CI,
  heuristic judge, 2017 staleness, single-brand/single-turn), next-week plan.
- `report/decision_log.md`: 15 entries (each: choice + rejected alternative).
- `README.md` (~1400 words, 10 sections): timed repro table, layout, taxonomy table, system +
  baselines, headline table tied to metrics files, golden summary, harness + agreement, failures /
  misleading / next-week, NOT-built, borrowings + submission note.
- `make report-check`: golden 150–250 (counter bug fixed — was counting header), artifacts exist,
  report length. Full `make eval` replay ~62 s (no `make` binary on this box — replayed as chained
  python commands).

## 11. Git history + identity

- Single squashed commit → rebuilt as **9 small classic commits** (scaffold → data scripts →
  taxonomy/prompts → baselines → agent → harness → data → results → docs), then README expansion.
- Identity saga: `author@local` → `Shiv Ratan Choudhary <shivbera45@gmail.com>` (profile scrape) →
  final `shivbera18 <164228363+shivbera18@users.noreply.github.com>` (from `gh auth status` +
  global gitconfig; local overrides removed). Remote `origin` =
  `github.com/shivbera18/hiver-support-agent.git`, pushed through the README commit.

## 12. Experiment loop (3 branches, PR-style review)

| branch | change | result | verdict |
|---|---|---|---|
| `experiment/sbert-retrieval` | MiniLM dense retrieval | torch DLL unloadable on this box; LSA fallback judge 2.04 < 2.40 | ❌ rejected |
| `experiment/intent-filtered-retrieval` | per-intent TF-IDF pools, shared weak map | judge 2.40 → **2.46**, ROUGE 0.105 → 0.119, intent/esc unchanged | ✅ **merged, default-on** |
| `experiment/complementnb-classifier` | ComplementNB vs LogReg | acc 0.675 < 0.705 (esc recall 0.97 not worth it) | ❌ rejected |

Reviewer subagent failed on its own output schema, so the 17-line diff was reviewed directly:
one cleanup enforced (single shared `_WEAK_KW` map, not three copies); 10× refit cost accepted
as batch-acceptable. Merge commit + adoption commit on `main`; docs updated to adopted numbers.
Experiment branches kept locally as paper trail.

## 13. LLM-fusion branch (in progress — BLOCKED)

Branch: `experiment/llm-fusion` (3 local commits; **not pushed — network died first**).
User supplied `.env` with `OPENAI_API_KEY` (56 chars) + `GEMINI_API_KEY` (53 chars); `.env` is
gitignored (verified untracked). New deps: `pydantic>=2.0`, `python-dotenv`, `google-genai`.

**Model findings (verified by probe, not assumed):**
- `gpt-5-mini` exists. Requires `max_completion_tokens` (`max_tokens` → 400). Reasoning model:
  burns ~800 completion tokens per reply; classify needs limit ≥1000 (100 truncated → 400 error).
  Per-row cost observed: ~500 in / ~150 out (classify), ~280 in / ~800 out (reply).
- `gemini-2.0-flash` and `gemini-2.5-flash` are **retired** (404 despite appearing in list API).
  `gemini-flash-latest` works but free-tier quota is 429-dead → Gemini gated behind
  `GEMINI_ENABLE` (unset = GPT-only). Fusion = **GPT primary + Gemini cascade** (asked only when
  GPT silent or conf < 0.7, saves ~70% of calls) + TF-IDF last resort.
- Pydantic `IntentOut` (`Literal[10 intents]`, `confidence ∈ [0,1]`) validates every classify
  output; `JudgeOut` (5 dims + `Literal[yes,no]`) validates every judge score. Judge order:
  Gemini → GPT → heuristic, each row tagged with scoring model.

**Loud-failure system (user requirement: nothing fails silently, tokens are expensive):**
- `results/llm_usage.json` ledger: per-call `LLM {stage} {model} OK/FAIL tok=in/out` lines,
  end-of-run `LLM SUMMARY`, `fallback_rows` counter, `ROW FALLBACK` / `ROW ERROR` per-row lines.
- Abort gates: agent refuses to write output if LLM success rate < 0.8 (`LLM_MIN_SUCCESS`);
  judge refuses mixed-scale scores if LLM-judged rows < 80%. **The gate fired correctly** on a
  7/400 mass-failure run (rate 0.02 → `SystemExit`, no output labelled as LLM results).
- Retries everywhere: GPT classify/reply (3×, backoff, explicit timeouts), Gemini (3×),
  judge GPT path (4×). Shared `OpenAI(max_retries=0)` singleton (SDK retries disabled in favor
  of ours). Threaded row loop (`AGENT_WORKERS`, default 8) with per-row try/except → canned +
  `ERROR:row-failed` instead of a crash. `run_eval.py` gained real BERTScore (`distilbert`)
  with skip-and-log fallback.

**Spend:** ~15k tokens total across probes (exact counts in ledger files). Full 200-row run
estimated ≈ 350k tokens (<$1 on mini). 5-row smoke: 10/10 calls OK, 0 fallbacks.

**Blocker (current): `WinError 10055` — this machine's network stack is down.** New sockets fail
to ALL hosts (OpenAI, Google, 8.8.8.8, github.com). Not quota/keys/code. 30-min watch confirmed
still down. Also means `git push` of the fusion branch is impossible right now.
**Resume when back:** `AGENT_WORKERS=12 python scripts/agent.py --in data/threads_sample.csv
--golden data/golden.csv --out results/predictions_fusion.jsonl` (~15 min), then judge-all,
agreement, comparison table, findings commit, PR.

## 14. Open items

1. [BLOCKED: network] Full 200-row fusion run → `results/predictions_fusion.jsonl`.
2. [BLOCKED: network] BERTScore eval + blinded LLM judge over all systems + fresh agreement.
3. [BLOCKED: network] Findings commit + PR (`experiment/llm-fusion` → `main`) + push.
4. Unblocked anytime: delete or merge the two rejected experiment branches; submit repo link via
   the Notion form (no email).
