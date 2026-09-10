# Data note
Primary: Kaggle thoughtvector/customer-support-on-twitter (~3M tweets). Get via `python scripts/download_data.py` (uses kagglehub; needs Kaggle credentials) or manually download `twcs.csv` from the Kaggle page into `data/twcs_raw.csv`.
Expected header: tweet_id,author_id,created_at,text,inbound,response_tweet_id,in_response_to_tweet_id (if different, adapt field map in build_threads.py only).
Subsample 200k rows seed=42. Threads: AppleSupport, 30k cap. Golden IDs excluded from train/retrieval pools (asserted).
Secondary: PolyAI/banking77 used ONLY as few-shot wording analogues in intent prompt, never trained on.
