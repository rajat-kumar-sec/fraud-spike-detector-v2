import pandas as pd
import numpy as np
import pickle
import json
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score
)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))


def parse_geo(s):
    lat, lon = s.split(",")
    return float(lat), float(lon)


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
def engineer_features(df):
    """
    Create features the ML model can learn from.
    This is THE most important step — garbage features = garbage model.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Time features ---
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_night"] = ((df["hour"] >= 1) & (df["hour"] <= 5)).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding for hour (helps model understand 23->0 is close)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # --- Amount features ---
    df["log_amount"] = np.log1p(df["amount"])
    df["amount Rounded"] = (df["amount"] // 1000) * 1000  # binned

    # --- Velocity features (per card) ---
    # How many transactions has this card done in last 1hr, 24hr?
    for window_name, window_mins in [("1h", 60), ("24h", 1440)]:
        col_name = f"card_txn_count_{window_name}"
        df[col_name] = 0

        for card, group in df.groupby("card_id"):
            indices = group.index.tolist()
            timestamps = group["timestamp"].values
            for i in range(len(indices)):
                start = timestamps[i] - np.timedelta64(window_mins, "m")
                count = np.sum((timestamps >= start) & (timestamps <= timestamps[i]))
                df.loc[indices[i], col_name] = count

    # --- Velocity features (per device) ---
    for window_name, window_mins in [("1h", 60), ("24h", 1440)]:
        col_name = f"device_txn_count_{window_name}"
        df[col_name] = 0

        for device, group in df.groupby("device_id"):
            indices = group.index.tolist()
            timestamps = group["timestamp"].values
            for i in range(len(indices)):
                start = timestamps[i] - np.timedelta64(window_mins, "m")
                count = np.sum((timestamps >= start) & (timestamps <= timestamps[i]))
                df.loc[indices[i], col_name] = count

    # --- Time since last transaction (per card) ---
    df["time_since_last_txn_card"] = 0.0
    for card, group in df.groupby("card_id"):
        indices = group.index.tolist()
        timestamps = group["timestamp"].values
        for i in range(1, len(indices)):
            gap = (timestamps[i] - timestamps[i - 1]) / np.timedelta64(1, "m")
            df.loc[indices[i], "time_since_last_txn_card"] = gap
    # First transaction per card gets a large value
    first_mask = df.groupby("card_id").cumcount() == 0
    df.loc[first_mask, "time_since_last_txn_card"] = 9999

    # --- Distance from previous transaction (per card) ---
    df["dist_from_last_txn"] = 0.0
    for card, group in df.groupby("card_id"):
        indices = group.index.tolist()
        for i in range(1, len(indices)):
            lat1, lon1 = parse_geo(group.loc[indices[i - 1], "geo_location"])
            lat2, lon2 = parse_geo(group.loc[indices[i], "geo_location"])
            dist = haversine(lat1, lon1, lat2, lon2)
            df.loc[indices[i], "dist_from_last_txn"] = dist
    df.loc[first_mask, "dist_from_last_txn"] = 0

    # --- Amount relative to card's average ---
    card_avg = df.groupby("card_id")["amount"].transform("mean")
    df["amount_vs_card_avg"] = df["amount"] / (card_avg + 1)

    # --- Rule engine outputs as features (from Step 2) ---
    # Re-apply quick versions of rules for features
    df["rule_burst"] = 0
    df["rule_geo_mismatch"] = 0
    df["rule_odd_hour"] = 0

    # Burst: count in 5-min window per card
    for card, group in df.groupby("card_id"):
        indices = group.index.tolist()
        timestamps = group["timestamp"].values
        for i in range(len(indices)):
            start = timestamps[i] - np.timedelta64(5, "m")
            count = np.sum((timestamps >= start) & (timestamps <= timestamps[i]))
            if count >= 5:
                df.loc[indices[i], "rule_burst"] = 1

    # Geo mismatch
    for card, group in df.groupby("card_id"):
        if len(group) < 2:
            continue
        indices = group.index.tolist()
        for i in range(len(indices) - 1):
            lat1, lon1 = parse_geo(group.loc[indices[i], "geo_location"])
            lat2, lon2 = parse_geo(group.loc[indices[i + 1], "geo_location"])
            dist = haversine(lat1, lon1, lat2, lon2)
            gap = (group.loc[indices[i + 1], "timestamp"] - group.loc[indices[i], "timestamp"]).total_seconds() / 60
            if dist > 500 and gap < 30:
                df.loc[indices[i], "rule_geo_mismatch"] = 1
                df.loc[indices[i + 1], "rule_geo_mismatch"] = 1

    # Odd hour
    df.loc[(df["hour"] >= 1) & (df["hour"] <= 5) & (df["amount"] >= 15000), "rule_odd_hour"] = 1

    return df


# ─────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "amount", "log_amount", "hour", "day_of_week", "is_night", "is_weekend",
    "hour_sin", "hour_cos",
    "card_txn_count_1h", "card_txn_count_24h",
    "device_txn_count_1h", "device_txn_count_24h",
    "time_since_last_txn_card", "dist_from_last_txn",
    "amount_vs_card_avg",
    "rule_burst", "rule_geo_mismatch", "rule_odd_hour",
]


def train_model(df):
    """
    Train a Gradient Boosting classifier with class balancing.
    Returns trained model, scaler, metrics, and test data.
    """
    X = df[FEATURE_COLS].values
    y = df["is_fraud"].values

    # Stratified split (preserves fraud ratio in train/test)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index.values, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train: {len(X_train)} samples ({y_train.sum()} fraud)")
    print(f"Test:  {len(X_test)} samples ({y_test.sum()} fraud)\n")

    # --- Model 1: Random Forest ---
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        class_weight="balanced",  # handles imbalance automatically
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train_scaled, y_train)

    # --- Model 2: Gradient Boosting ---
    print("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        subsample=0.8,
        random_state=42,
    )
    gb.fit(X_train_scaled, y_train)

    # --- Evaluate both ---
    models = {"Random Forest": rf, "Gradient Boosting": gb}
    best_model = None
    best_f1 = 0

    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="binary"
        )
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n{'=' * 50}")
        print(f"  {name}")
        print(f"{'=' * 50}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall:    {recall:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  AUC-ROC:   {auc:.3f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
        print(f"    FN={cm[1][0]}  TP={cm[1][1]}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = (name, model)

    print(f"\n>>> Best model: {best_model[0]} (F1={best_f1:.3f})")

    # --- Feature importance (from best model) ---
    importances = best_model[1].feature_importances_
    feat_imp = sorted(
        zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True
    )
    print(f"\n  Top 10 Features:")
    for feat, imp in feat_imp[:10]:
        bar = "#" * int(imp * 100)
        print(f"    {feat:30s} {imp:.3f} {bar}")

    return best_model[1], scaler, best_model[0], {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(best_f1, 3),
        "auc_roc": round(auc, 3),
    }, X_test, y_test, idx_test


def save_artifacts(model, scaler, metrics, feature_cols):
    """Save model, scaler, and metadata for dashboard use."""
    # Save model + scaler
    with open("fraud_model.pkl", "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "features": feature_cols}, f)

    # Save metrics as JSON for dashboard
    with open("model_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nSaved: fraud_model.pkl, model_metrics.json")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load data from Step 1
    print("Loading data...")
    df = pd.read_csv("transactions.csv", parse_dates=["timestamp"])
    print(f"Loaded {len(df)} transactions\n")

    # Feature engineering
    print("Engineering features...")
    df = engineer_features(df)
    print(f"Created {len(FEATURE_COLS)} features\n")

    # Train
    model, scaler, model_name, metrics, X_test, y_test, idx_test = train_model(df)

    # Save
    save_artifacts(model, scaler, metrics, FEATURE_COLS)

    # Save test predictions for dashboard
    X_test_scaled = scaler.transform(X_test)
    test_df = df.loc[idx_test].copy()
    test_df["ml_prediction"] = model.predict(X_test_scaled)
    test_df["ml_probability"] = model.predict_proba(X_test_scaled)[:, 1]
    test_df.to_csv("test_predictions.csv", index=False)
    print(f"Saved test predictions ({len(test_df)} rows) to test_predictions.csv")
