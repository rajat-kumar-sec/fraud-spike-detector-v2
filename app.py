import pandas as pd
import numpy as np
import pickle
import json
from flask import Flask, jsonify, send_from_directory
from ml_model import engineer_features, FEATURE_COLS
from rule_engine import run_rule_engine, evaluate_rules
from explainer import explain_transaction

app = Flask(__name__, static_folder="static")

# ─────────────────────────────────────────────────────────────
# LOAD ARTIFACTS ON STARTUP
# ─────────────────────────────────────────────────────────────
print("Loading model artifacts...")
with open("fraud_model.pkl", "rb") as f:
    artifact = pickle.load(f)
    model = artifact["model"]
    scaler = artifact["scaler"]

with open("model_metrics.json") as f:
    model_metrics = json.load(f)

print("Loading and processing data...")
df_raw = pd.read_csv("transactions.csv", parse_dates=["timestamp"])
df = engineer_features(df_raw)

# Run rule engine on raw data
df_rules = run_rule_engine(df_raw.copy())

# Get ML predictions for all transactions
X_all = df[FEATURE_COLS].values
X_scaled = scaler.transform(X_all)
df["ml_prediction"] = model.predict(X_scaled)
df["ml_probability"] = model.predict_proba(X_scaled)[:, 1]

# Merge rule flags into main df
df["rule_burst"] = df_rules["rule_burst"].values
df["rule_geo_mismatch"] = df_rules["rule_geo_mismatch"].values
df["rule_odd_hour"] = df_rules["rule_odd_hour"].values
df["risk_score"] = df_rules["risk_score"].values

# Compute combined prediction (ML or rules)
df["combined_prediction"] = ((df["ml_prediction"] == 1) | (df["risk_score"] > 0)).astype(int)

print(f"Ready. {len(df)} transactions loaded.\n")


# ─────────────────────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/summary")
def api_summary():
    """Overall dashboard summary stats."""
    total = len(df)
    fraud = int(df["is_fraud"].sum())
    flagged_ml = int(df["ml_prediction"].sum())
    flagged_rules = int((df["risk_score"] > 0).sum())
    flagged_combined = int(df["combined_prediction"].sum())

    return jsonify({
        "total_transactions": total,
        "actual_fraud": fraud,
        "actual_normal": total - fraud,
        "flagged_by_ml": flagged_ml,
        "flagged_by_rules": flagged_rules,
        "flagged_combined": flagged_combined,
        "model_metrics": model_metrics,
    })


@app.route("/api/transactions")
def api_transactions():
    """All transactions with predictions (for the table)."""
    cols = [
        "transaction_id", "amount", "timestamp", "card_id", "device_id",
        "ip_address", "merchant_id", "geo_location", "is_fraud",
        "ml_prediction", "ml_probability", "rule_burst", "rule_geo_mismatch",
        "rule_odd_hour", "risk_score", "combined_prediction",
    ]
    out = df[cols].copy()
    out["timestamp"] = out["timestamp"].astype(str)
    out["ml_probability"] = out["ml_probability"].round(4)
    return jsonify(out.to_dict(orient="records"))


@app.route("/api/transaction/<txn_id>")
def api_transaction_detail(txn_id):
    """Single transaction with Claude explanation."""
    row = df[df["transaction_id"] == txn_id]
    if row.empty:
        return jsonify({"error": "Transaction not found"}), 404

    txn = row.iloc[0].to_dict()
    txn["timestamp"] = str(txn["timestamp"])

    # Get explanation
    explanation = explain_transaction(txn)
    txn["explanation"] = explanation

    return jsonify(txn)


@app.route("/api/flagged")
def api_flagged():
    """Only transactions flagged by ML or rules."""
    flagged = df[df["combined_prediction"] == 1].copy()
    cols = [
        "transaction_id", "amount", "timestamp", "card_id", "device_id",
        "merchant_id", "geo_location", "is_fraud",
        "ml_prediction", "ml_probability", "rule_burst", "rule_geo_mismatch",
        "rule_odd_hour", "risk_score",
    ]
    out = flagged[cols].copy()
    out["timestamp"] = out["timestamp"].astype(str)
    out["ml_probability"] = out["ml_probability"].round(4)
    return jsonify(out.to_dict(orient="records"))


@app.route("/api/metrics")
def api_metrics():
    """Model + rule metrics for dashboard charts."""
    # ML metrics
    ml_tp = ((df["ml_prediction"] == 1) & (df["is_fraud"] == 1)).sum()
    ml_fp = ((df["ml_prediction"] == 1) & (df["is_fraud"] == 0)).sum()
    ml_fn = ((df["ml_prediction"] == 0) & (df["is_fraud"] == 1)).sum()
    ml_tn = ((df["ml_prediction"] == 0) & (df["is_fraud"] == 0)).sum()

    # Rule metrics
    rules_eval = evaluate_rules(df_raw)

    return jsonify({
        "ml_confusion": {"TP": int(ml_tp), "FP": int(ml_fp), "FN": int(ml_fn), "TN": int(ml_tn)},
        "rules": rules_eval,
        "model_metrics": model_metrics,
    })


@app.route("/api/risk-distribution")
def api_risk_distribution():
    """Distribution of ML probability scores for charting."""
    bins = list(np.arange(0, 1.05, 0.05))
    hist_fraud, _ = np.histogram(df[df["is_fraud"] == 1]["ml_probability"], bins=bins)
    hist_normal, _ = np.histogram(df[df["is_fraud"] == 0]["ml_probability"], bins=bins)
    labels = [f"{b:.2f}-{b+0.05:.2f}" for b in bins[:-1]]

    return jsonify({
        "labels": labels,
        "fraud": hist_fraud.tolist(),
        "normal": hist_normal.tolist(),
    })


@app.route("/api/time-series")
def api_time_series():
    """Transactions over time, split by fraud/normal."""
    df_ts = df.copy()
    df_ts["date"] = df_ts["timestamp"].dt.date.astype(str)
    daily = df_ts.groupby(["date", "is_fraud"]).size().unstack(fill_value=0)
    daily = daily.reset_index()
    if 0 in daily.columns and 1 in daily.columns:
        daily = daily.rename(columns={0: "normal", 1: "fraud"})
    else:
        daily["normal"] = 0
        daily["fraud"] = 0

    return jsonify({
        "dates": daily["date"].tolist(),
        "normal": daily["normal"].tolist(),
        "fraud": daily["fraud"].tolist(),
    })


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting Fraud Spike Detector on http://localhost:3000")
    app.run(debug=False, port=3000)
