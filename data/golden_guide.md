# Golden guide (1 page)
Intent definitions + 2 examples each + tie-breaks (see intent prompt). Multi-intent -> action-requiring intent. Sarcasm/praise+complaint -> complaint. Non-English/URL-only/image-only/short-unclear -> other_offtopic + escalate.
Escalate reasons: LOWCONF:x RISK:x SAFETY:x MISSING-INFO OK:...
## Sampling/labelling note
Sampled 2017-10/11 tweets (twcs span), labelled 2026-09-10 from threads_sample (AppleSupport, time-ordered): 15/intent weak-label buckets x10=150 + 30 uniform random + 20 adversarial (short/caps/sarcasm/non-English/multi-intent) = 200. Dedup max-3 copies. Single-author labels; 50-item 48h re-label subset for intra-annotator kappa. Skew: English-heavy, excludes deleted tweets. Leakage: golden IDs asserted out of train/retrieval pools.
Reply references are author-written action templates per intent (reviewed per row); setup rows fixed post-review ({step} placeholder removed). Pass-2 re-label of 50 rows flipped 1 label (R7 app_store->billing per charge tie-break).
