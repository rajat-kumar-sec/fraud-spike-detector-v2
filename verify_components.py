import sys
sys.stdout.reconfigure(encoding="utf-8")

from ml_model import engineer_features, FEATURE_COLS
from rule_engine import run_rule_engine, evaluate_rules
from explainer import explain_transaction
print("[OK] All imports work")

import pandas as pd, pickle

df = pd.read_csv("transactions.csv", parse_dates=["timestamp"])
print(f"[OK] Loaded {len(df)} rows")

with open("fraud_model.pkl", "rb") as f:
    art = pickle.load(f)
model = art["model"]
print(f"[OK] Model loaded: {type(model).__name__}")

# Test explanation
txn = {
    "transaction_id": "TXN000001",
    "amount": 50000,
    "timestamp": "2026-08-15 03:00:00",
    "card_id": "C123456",
    "device_id": "D99999",
    "geo_location": "19.0,72.8",
    "ml_probability": 0.95,
    "rule_burst": 0,
    "rule_geo_mismatch": 0,
    "rule_odd_hour": 1,
}
expl = explain_transaction(txn)
print(f"[OK] Explanation: {expl[:150]}...")
print("\nAll components verified. Ready for dashboard.")
