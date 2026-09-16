# Golden guide
## Intent definitions (2 examples each from the golden set)
- setup_howto (setup/pairing/how-to): "Just got AirPods, how do I pair to iPhone?"; "Is there a supported splitter for headphones and charging on iPhone 7+?"
- battery_performance (drain/slow/heat): "battery is draining faster after updating to iOS 11.0.2 in iPhone 6s"; "is there a way to check my battery health?"
- software_update (update stuck/failed): "updated my iphone 7 to ios 11.0.2 and it keeps freezing"; "I hate that everytime a new phone releases the previous versions start glitching"
- account_icloud (ID/login/sync/password): "I am unable to login"; "No this is my 20th try, I have restarted, force restarted, turned off iCloud"
- hardware_damage_repair (damage/repair status): "Can I get a new screen?"; "Thanks v much Lewis ... That does not fix my broken laptop before I head off overseas"
- warranty_applecare (coverage/AppleCare): "It's out of warranty, what are my options?"; "Is there a way to buy extended warranty for a MacBook Pro?"
- app_store_itunes (download/purchase/subscription): "Can't access the App Store for a week now HELP"; "whenever I BUY a song on iTunes it charges me then tells me I haven't purchased it"
- connectivity (wifi/bluetooth/cellular/AirDrop): "WiFi, I'm using iPad."; "It doesn't happen if I just switch back and forth. But every time I get in my car (Bluetooth enabled)..."
- billing_refund (charge/refund/receipt): "Noo but I logged in to Apple Music again and they have charged me again"; "whenever I BUY a song on iTunes it charges me then tells me I haven't purchased it" (charge needs action -> billing per tie-break)
- other_offtopic (praise/spam/non-support/incomprehensible): "Yay! Thank you"; "how do I find my booking?" (non-Apple content)
## Tie-breaks (enforced by all classifiers)
Multi-intent -> the one requiring action. Sarcasm/praise + complaint -> complaint. Non-English/URL-only/image-only/short-unclear -> other_offtopic + escalate. Charge words + store context -> billing_refund.
Escalate reasons: LOWCONF:x RISK:x SAFETY:x MISSING-INFO OK:...
## Sampling/labelling note
Sampled 2017-10/11 tweets (twcs span), labelled 2026-09-10 from threads_sample (AppleSupport, time-ordered): 15/intent weak-label buckets x10=150 + 30 uniform random + 20 adversarial (short/caps/sarcasm/non-English/multi-intent) = 200. Dedup max-3 copies. Single-author labels; 50-item re-label subset (kappa 0.97, same-session upper bound) flipped 1 label (R7 app_store->billing per charge tie-break). Skew: English-heavy, excludes deleted tweets, setup_howto thin (n=6). Leakage: golden IDs asserted out of train/retrieval pools.
Reply references are author-written action templates per intent (reviewed per row); setup rows fixed post-review ({step} placeholder removed).
