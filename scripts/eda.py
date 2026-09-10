"""EDA: brand volumes, lengths, non-english proxy, orphan %. --help/--sample-n/--seed/--out."""
import argparse, pandas as pd, re
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/twcs_raw.csv")
    ap.add_argument("--out", default="results/eda_summary.csv")
    ap.add_argument("--sample-n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    df = pd.read_csv(a.inp, low_memory=False)
    print("columns:", list(df.columns))
    # normalize likely column names
    lc = {c.lower(): c for c in df.columns}
    def col(*names):
        for n in names:
            if n in lc: return lc[n]
        return None
    c_text = col("text") or "text"; c_inb = col("inbound") or "inbound"
    c_auth = col("author_id")
    brand = c_auth if c_auth else c_text
    is_in = df[c_inb] == True if df[c_inb].dtype == bool else df[c_inb].astype(str).str.lower().isin(["true","1"])
    inb, outb = df[is_in], df[~is_in]
    print(f"rows={len(df)} inbound={len(inb)} outbound={len(outb)} span_check: run describe on created_at if present")
    top = outb[c_auth].value_counts().head(15)
    print(top.to_string())
    inb_txt = inb[c_text].astype(str)
    med_len = inb_txt.str.len().median()
    non_ascii = (inb_txt.apply(lambda s: sum(ord(ch) > 127 for ch in s) / max(len(s),1) > 0.3)).mean()
    c_resp = col("in_response_to_tweet_id")
    orphan = (inb[c_resp].isna().mean() if c_resp else float("nan"))
    print(f"median_len={med_len:.0f} non_ascii_proxy={non_ascii:.3f} orphan={orphan:.3f}")
    top.to_csv(a.out, header=["inbound_count"])
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top.head(15).plot(kind="barh"); plt.gca().invert_yaxis(); plt.tight_layout()
        plt.savefig("results/brand_volume.png"); print("chart results/brand_volume.png")
    except Exception as e: print("chart skipped:", e)
if __name__ == "__main__": main()
