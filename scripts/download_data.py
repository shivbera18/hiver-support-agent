"""Download twcs via kagglehub, subsample. --help/--sample-n/--seed/--out."""
import argparse, sys, pandas as pd
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/twcs_raw.csv")
    ap.add_argument("--sample-n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    try:
        import kagglehub
        p = kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
        import os, glob
        csvs = glob.glob(os.path.join(p, "**/*.csv"), recursive=True)
        if not csvs: print("no csv in kagglehub download", file=sys.stderr); sys.exit(1)
        csvs = sorted(csvs, key=os.path.getsize, reverse=True)
        print(f"using largest csv: {csvs[0]} ({os.path.getsize(csvs[0])} bytes)")
        df = pd.read_csv(csvs[0], low_memory=False)
    except Exception as e:
        print(f"auto-download failed ({e}). Download manually: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter -> data/twcs_raw.csv. See data/README_DATA.md", file=sys.stderr)
        sys.exit(1)
    if len(df) > a.sample_n:
        df = df.sample(n=a.sample_n, random_state=a.seed).sort_index()
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out} rows={len(df)} cols={list(df.columns)}")
if __name__ == "__main__": main()
