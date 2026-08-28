import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))  # Earth radius ~6371 km


def parse_geo(geo_str):
    """Parse 'lat,lon' string into tuple of floats."""
    lat, lon = geo_str.split(",")
    return float(lat), float(lon)


# ─────────────────────────────────────────────────────────────
# RULE 1: Rapid Burst Detection
# Same card or device making many transactions in a short window
# ─────────────────────────────────────────────────────────────
def rule_rapid_burst(df, card_threshold=5, device_threshold=5, window_minutes=5):
    """
    Flag transactions where the same card or device has >= N
    transactions within a sliding window of T minutes.

    How it works:
    - For each transaction, look back at all transactions from the
      same card_id within the last `window_minutes`.
    - If count >= threshold, flag ALL of them as suspicious.
    - Same logic for device_id.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["burst_flag_card"] = 0
    df["burst_flag_device"] = 0

    # Group by card
    for card, group in df.groupby("card_id"):
        idx = group.index.tolist()
        timestamps = group["timestamp"].values
        for i in range(len(idx)):
            # Count how many txns from this card are within window
            window_start = timestamps[i] - np.timedelta64(window_minutes, "m")
            count = np.sum((timestamps >= window_start) & (timestamps <= timestamps[i]))
            if count >= card_threshold:
                df.loc[idx[i], "burst_flag_card"] = 1

    # Group by device
    for device, group in df.groupby("device_id"):
        idx = group.index.tolist()
        timestamps = group["timestamp"].values
        for i in range(len(idx)):
            window_start = timestamps[i] - np.timedelta64(window_minutes, "m")
            count = np.sum((timestamps >= window_start) & (timestamps <= timestamps[i]))
            if count >= device_threshold:
                df.loc[idx[i], "burst_flag_device"] = 1

    df["rule_burst"] = ((df["burst_flag_card"] == 1) | (df["burst_flag_device"] == 1)).astype(int)
    df = df.drop(columns=["burst_flag_card", "burst_flag_device"])
    return df


# ─────────────────────────────────────────────────────────────
# RULE 2: Geo-Mismatch Detection
# Same card appearing in distant cities within short time
# ─────────────────────────────────────────────────────────────
def rule_geo_mismatch(df, distance_km=500, window_minutes=30):
    """
    Flag transactions where the same card is used in two locations
    that are > distance_km apart within window_minutes.

    How it works:
    - For each card, compare every pair of consecutive transactions.
    - Calculate haversine distance between their geo_locations.
    - If distance > threshold AND time gap < window, flag both.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["geo_mismatch_flag"] = 0

    for card, group in df.groupby("card_id"):
        if len(group) < 2:
            continue
        indices = group.index.tolist()
        for i in range(len(indices) - 1):
            lat1, lon1 = parse_geo(group.loc[indices[i], "geo_location"])
            lat2, lon2 = parse_geo(group.loc[indices[i + 1], "geo_location"])
            dist = haversine(lat1, lon1, lat2, lon2)
            time_gap = (
                group.loc[indices[i + 1], "timestamp"] - group.loc[indices[i], "timestamp"]
            ).total_seconds() / 60

            if dist > distance_km and time_gap < window_minutes:
                df.loc[indices[i], "geo_mismatch_flag"] = 1
                df.loc[indices[i + 1], "geo_mismatch_flag"] = 1

    df["rule_geo_mismatch"] = df["geo_mismatch_flag"]
    df = df.drop(columns=["geo_mismatch_flag"])
    return df


# ─────────────────────────────────────────────────────────────
# RULE 3: Odd-Hour High-Value Detection
# Transactions between 1-5 AM with amount above threshold
# ─────────────────────────────────────────────────────────────
def rule_odd_hour_high_value(df, start_hour=1, end_hour=5, min_amount=15000):
    """
    Flag transactions that happen during unusual hours (1-5 AM)
    AND have high amounts.

    Simple but effective: fraudsters often transact at night
    when legitimate activity is low.
    """
    df = df.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["rule_odd_hour"] = (
        (df["hour"] >= start_hour) & (df["hour"] <= end_hour) & (df["amount"] >= min_amount)
    ).astype(int)
    df = df.drop(columns=["hour"])
    return df


# ─────────────────────────────────────────────────────────────
# COMBINE RULES: Final prediction
# ─────────────────────────────────────────────────────────────
def run_rule_engine(df):
    """
    Run all rules and produce a final prediction.
    Returns:
    - df with rule columns + 'rule_prediction' (0/1)
    - dict of per-rule and combined metrics
    """
    print("Running rule engine...")

    # Apply rules
    df = rule_rapid_burst(df)
    print("  [OK] Rapid burst rule applied")

    df = rule_geo_mismatch(df)
    print("  [OK] Geo-mismatch rule applied")

    df = rule_odd_hour_high_value(df)
    print("  [OK] Odd-hour high-value rule applied")

    # Combine: flag if ANY rule fires
    df["rule_prediction"] = (
        (df["rule_burst"] == 1) |
        (df["rule_geo_mismatch"] == 1) |
        (df["rule_odd_hour"] == 1)
    ).astype(int)

    # Risk score: count of rules that fired (0, 1, 2, or 3)
    df["risk_score"] = (
        df["rule_burst"] + df["rule_geo_mismatch"] + df["rule_odd_hour"]
    )

    return df


def evaluate_rules(df):
    """Calculate precision, recall, F1 for each rule and combined."""
    true = df["is_fraud"]
    results = {}

    rules = {
        "Rapid Burst": "rule_burst",
        "Geo Mismatch": "rule_geo_mismatch",
        "Odd-Hour High-Value": "rule_odd_hour",
        "Combined (any rule)": "rule_prediction",
    }

    for name, col in rules.items():
        pred = df[col]
        tp = ((pred == 1) & (true == 1)).sum()
        fp = ((pred == 1) & (true == 0)).sum()
        fn = ((pred == 0) & (true == 1)).sum()
        tn = ((pred == 0) & (true == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[name] = {
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "F1": round(f1, 3),
        }

    return results


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("transactions.csv", parse_dates=["timestamp"])
    print(f"Loaded {len(df)} transactions\n")

    # Run engine
    df = run_rule_engine(df)

    # Evaluate
    metrics = evaluate_rules(df)

    print("\n" + "=" * 65)
    print("  RULE-BASED DETECTION RESULTS")
    print("=" * 65)
    for rule_name, m in metrics.items():
        print(f"\n  {rule_name}:")
        print(f"    TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}")
        print(f"    Precision={m['Precision']}  Recall={m['Recall']}  F1={m['F1']}")
    print("\n" + "=" * 65)

    # Save results
    df.to_csv("transactions_with_rules.csv", index=False)
    print(f"\nSaved to: transactions_with_rules.csv")
    print(f"Columns: {list(df.columns)}")
