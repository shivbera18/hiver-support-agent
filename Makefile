setup:
	pip install -r requirements.txt

data:
	python scripts/download_data.py --out data/twcs_raw.csv --sample-n 200000 --seed 42
	python scripts/eda.py --in data/twcs_raw.csv --out results/eda_summary.csv
	python scripts/build_threads.py --in data/twcs_raw.csv --brand AppleSupport --out data/threads_sample.csv

eval:
	python scripts/baseline_trivial.py --in data/threads_sample.csv --golden data/golden.csv --out results/predictions_baseline_trivial.jsonl
	python scripts/baseline_tfidf.py --in data/threads_sample.csv --golden data/golden.csv --out results/predictions_baseline_tfidf.jsonl
	python scripts/agent.py --in data/threads_sample.csv --golden data/golden.csv --out results/predictions_main.jsonl
	python scripts/run_eval.py --golden data/golden.csv --pred results/predictions_baseline_trivial.jsonl --out results/metrics_trivial.json
	python scripts/run_eval.py --golden data/golden.csv --pred results/predictions_baseline_tfidf.jsonl --out results/metrics_tfidf.json
	python scripts/run_eval.py --golden data/golden.csv --pred results/predictions_main.jsonl --out results/metrics.json
	python scripts/judge.py --pred results/predictions_main.jsonl --golden data/golden.csv --out results/judge_scores.csv
	python scripts/judge.py --pred results/predictions_baseline_tfidf.jsonl --golden data/golden.csv --out results/judge_tfidf.csv
	python scripts/judge.py --pred results/predictions_baseline_trivial.jsonl --golden data/golden.csv --out results/judge_trivial.csv
	python scripts/agreement.py --judge results/judge_scores.csv --out results/agreement.json
	python scripts/failure_mine.py --metrics results/metrics.json --judge results/judge_scores.csv --out results/failures.md

report-check:
	python -c "import csv;rows=list(csv.DictReader(open('data/golden.csv',encoding='utf-8')));assert 150<=len(rows)<=250, len(rows);print('golden count OK:',len(rows))"
	python -c "import json;assert __import__('pathlib').Path('results/metrics.json').exists();assert __import__('pathlib').Path('results/agreement.json').exists();print('artifacts OK')"
	python -c "words=len(open('report/report.md',encoding='utf-8').read().split());assert words<=3000,words;print('report words OK:',words)"

eval-llm:
	AGENT_WORKERS=12 python scripts/agent.py --in data/threads_sample.csv --golden data/golden.csv --out results/predictions_fusion.jsonl
	python scripts/run_eval.py --golden data/golden.csv --pred results/predictions_fusion.jsonl --out results/metrics_fusion.json
	python scripts/judge.py --pred results/predictions_fusion.jsonl --golden data/golden.csv --out results/judge_fusion.csv
	python scripts/judge.py --pred results/predictions_main.jsonl --golden data/golden.csv --out results/judge_main_llm.csv
	python scripts/judge.py --pred results/predictions_baseline_tfidf.jsonl --golden data/golden.csv --out results/judge_tfidf_llm.csv
	python scripts/judge.py --pred results/predictions_baseline_trivial.jsonl --golden data/golden.csv --out results/judge_trivial_llm.csv
	python scripts/agreement.py --judge results/judge_fusion.csv --out results/agreement_fusion.json
