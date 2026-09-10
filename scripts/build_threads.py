"""One row per inbound msg with 1-hop brand+customer context and next brand reply. --help/--sample-n/--seed/--out."""
import argparse, re, pandas as pd
MAXLEN=500
def norm(s):
    s = re.sub(r"^@\w+\s*", "", str(s))
    s = re.sub(r"http\S+", "<URL>", s)
    s = s.strip()
    if len(s) > MAXLEN: s = s[:MAXLEN] + " <TRUNC>"
    return s
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/twcs_raw.csv")
    ap.add_argument("--brand", default="AppleSupport")
    ap.add_argument("--out", default="data/threads_sample.csv")
    ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = pd.read_csv(a.inp, low_memory=False)
    lc = {c.lower(): c for c in df.columns}
    T, AU, INB, RT, IR = lc.get("text","text"), lc.get("author_id","author_id"), lc.get("inbound","inbound"), lc.get("response_tweet_id","response_tweet_id"), lc.get("in_response_to_tweet_id","in_response_to_tweet_id")
    TID = lc.get("tweet_id","tweet_id"); CA = lc.get("created_at", None)
    is_in = df[INB] == True if df[INB].dtype == bool else df[INB].astype(str).str.lower().isin(["false","0"])
    inb_all = df[is_in].copy()
    outb_brand = df[(~is_in) & (df[AU].astype(str).str.lower() == a.brand.lower())].copy()
    brand_parent_ids = set(outb_brand[IR].astype(str)) | set(outb_brand[TID].astype(str))
    inb = inb_all[(inb_all[TID].astype(str).isin(brand_parent_ids)) | (inb_all[RT].astype(str).isin(brand_parent_ids)) | (inb_all[T].astype(str).str.contains("@" + a.brand, case=False, na=False))].copy()
    print(f"brand={a.brand} brand_outbound={len(outb_brand)} brand_inbound={len(inb)}")
    if len(inb) == 0: print("WARNING: no brand inbound; check brand name"); inb = inb_all.head(30000)
    # brand reply lookup: tweet whose IR points at inbound TID
    outb = df[df[INB] == False] if df[INB].dtype == bool else df[df[INB].astype(str).str.lower().isin(["false","0"])]
    reply_by_id = {str(r[TID]): str(r[T]) for _, r in outb.iterrows()}
    by_parent = {}
    for _, r in outb.iterrows():
        if pd.notna(r[IR]): by_parent.setdefault(str(r[IR]).split(".")[0], str(r[T]))
    by_id = {str(r[TID]): r for _, r in df.iterrows()}
    rows = []
    for _, r in inb.iterrows():
        t = norm(r[T])
        if not t: continue
        par_b, par_c = "", ""
        p = by_id.get(str(r[IR])) if IR in df.columns and pd.notna(r[IR]) else None
        if p is not None:
            pt = norm(p[T]); par_b = pt if str(p[AU]).lower() == a.brand.lower() else ""; par_c = "" if str(p[AU]).lower() == a.brand.lower() else pt
        rows.append({"msg_id": str(r[TID]), "brand": a.brand, "customer_text": t, "parent_brand_text": par_b, "parent_customer_text": par_c, "created_at": str(r[CA]) if CA else "", "brand_reply_text": (lambda rt: reply_by_id.get(str(rt).split(",")[0].split(".")[0], by_parent.get(str(r[TID]).split(".")[0], "")))(r[RT]) if pd.notna(r[RT]) else by_parent.get(str(r[TID]).split(".")[0], "")})
    d = pd.DataFrame(rows)
    d = d[d.customer_text.str.len() > 0]
    # dedup: keep max 3 copies
    d["_h"] = d.customer_text.str.lower().str.strip()
    d = d.groupby("_h", group_keys=False).head(3).drop(columns=["_h"])
    d = d.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    if len(d) > 30000: d = d.head(30000)
    d.to_csv(a.out, index=False)
    print(f"wrote {a.out} rows={len(d)}")
if __name__ == "__main__": main()
