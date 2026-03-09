import argparse # allows the script to accept input/output file paths from the command line
import numpy as np
import pandas as pd
from pathlib import Path

#creates time-based and cyclical features from the timestamp.
def add_time_features(df):
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.copy() # avoids modifying the original DataFrame unintentionally
    df["timestamp"] = ts # converts timestamp column into a datetime format

    # Extract time components
    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["day"] = ts.dt.day
    df["hour"] = ts.dt.hour
    df["dayofyear"] = ts.dt.dayofyear
    df["weekday"] = ts.dt.weekday

    # Create cyclical (sin/cos) features of time
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    df["doy_sin"] = np.sin(2*np.pi*df["dayofyear"]/365.25)
    df["doy_cos"] = np.cos(2*np.pi*df["dayofyear"]/365.25)
    
    #Return enhanced dataframe
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)  # processed CSV
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    #Removes duplicate rows
    df = df.drop_duplicates()
    #Drops rows with invalid timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # convert key columns to numeric
    for c in ["ALLSKY_SFC_SW_DWN", "T2M", "RH2M", "WS2M", "lat", "lon"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # domain sanity check (solar irradiance cannot be negative)
    df["ALLSKY_SFC_SW_DWN"] = df["ALLSKY_SFC_SW_DWN"].clip(lower=0)

    # add engineered time features
    df = add_time_features(df)

    # one-hot encode city (6 cities)
    df = pd.get_dummies(df, columns=["city"], prefix="city", drop_first=False)

    # missing value handling: median fill (simple + robust)
    df = df.fillna(df.median(numeric_only=True))
    # sorts data chornologically
    df = df.sort_values("timestamp").reset_index(drop=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Saved processed data: {out} rows={len(df):,}, cols={df.shape[1]}")

if __name__ == "__main__":
    main()
