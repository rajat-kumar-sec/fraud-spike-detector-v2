import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import timedelta

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

NUM_TRANSACTIONS = 1500
FRAUD_RATE = 0.09  # ~9% fraud

# --- Configuration ---
MERCHANTS = [f"M{i:04d}" for i in range(1, 51)]  # 50 merchants
IP_POOL = [fake.ipv4() for _ in range(200)]  # pool of IPs
DEVICES = [f"D{random.randint(10000,99999)}" for _ in range(300)]
CARDS = [f"C{random.randint(100000,999999)}" for _ in range(400)]

# Real Indian metro coordinates with some jitter
LOCATIONS = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.7041, 77.1025),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Ahmedabad": (23.0225, 72.5714),
}
LOC_NAMES = list(LOCATIONS.keys())


def jitter_coords(base, spread=0.5):
    return (
        round(base[0] + np.random.uniform(-spread, spread), 4),
        round(base[1] + np.random.uniform(-spread, spread), 4),
    )


def generate_normal_transactions(n):
    """Generate normal (non-fraud) transactions."""
    rows = []
    for _ in range(n):
        city = random.choice(LOC_NAMES)
        lat, lon = jitter_coords(LOCATIONS[city])
        raw_p = [
            0.01 if h in range(1, 5) else 0.04 if h in range(5, 8) else 0.06
            for h in range(24)
        ]
        p = np.array(raw_p) / sum(raw_p)
        hour = np.random.choice(range(24), p=p)
        ts = fake.date_time_between(start_date="-30d", end_date="now").replace(
            hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
        )
        rows.append(
            {
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.lognormal(mean=6.5, sigma=1.0), 2),
                "timestamp": ts,
                "card_id": random.choice(CARDS),
                "device_id": random.choice(DEVICES),
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat},{lon}",
                "is_fraud": 0,
            }
        )
    return rows


def generate_fraud_burst(n_bursts=8, txns_per_burst=6):
    """Fraud pattern 1: Rapid transactions from same device/card in short window."""
    rows = []
    for _ in range(n_bursts):
        device = random.choice(DEVICES)
        card = random.choice(CARDS)
        city = random.choice(LOC_NAMES)
        lat, lon = jitter_coords(LOCATIONS[city])
        base_time = fake.date_time_between(start_date="-30d", end_date="now")
        merchant = random.choice(MERCHANTS)
        for j in range(txns_per_burst):
            ts = base_time + timedelta(seconds=random.randint(5, 120))
            rows.append(
                {
                    "transaction_id": f"TXN{random.randint(100000, 999999)}",
                    "amount": round(np.random.uniform(5000, 50000), 2),
                    "timestamp": ts,
                    "card_id": card,
                    "device_id": device,
                    "ip_address": random.choice(IP_POOL),
                    "merchant_id": merchant,
                    "geo_location": f"{lat},{lon}",
                    "is_fraud": 1,
                }
            )
    return rows


def generate_geo_mismatch(n=6):
    """Fraud pattern 2: Same card used in far-apart cities within minutes."""
    rows = []
    for _ in range(n):
        card = random.choice(CARDS)
        device = random.choice(DEVICES)
        city1, city2 = random.sample(LOC_NAMES, 2)
        lat1, lon1 = jitter_coords(LOCATIONS[city1])
        lat2, lon2 = jitter_coords(LOCATIONS[city2])
        base_time = fake.date_time_between(start_date="-30d", end_date="now")
        # Two transactions ~10 min apart but different cities (impossible travel)
        rows.append(
            {
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(8000, 40000), 2),
                "timestamp": base_time,
                "card_id": card,
                "device_id": device,
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat1},{lon1}",
                "is_fraud": 1,
            }
        )
        rows.append(
            {
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(8000, 40000), 2),
                "timestamp": base_time + timedelta(minutes=random.randint(5, 20)),
                "card_id": card,
                "device_id": device,
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat2},{lon2}",
                "is_fraud": 1,
            }
        )
    return rows


def generate_odd_hour_high_value(n=8):
    """Fraud pattern 3: High-value transactions at unusual hours (1-5 AM)."""
    rows = []
    for _ in range(n):
        city = random.choice(LOC_NAMES)
        lat, lon = jitter_coords(LOCATIONS[city])
        hour = random.choice([1, 2, 3, 4])
        ts = fake.date_time_between(start_date="-30d", end_date="now").replace(
            hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
        )
        rows.append(
            {
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(15000, 80000), 2),
                "timestamp": ts,
                "card_id": random.choice(CARDS),
                "device_id": random.choice(DEVICES),
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat},{lon}",
                "is_fraud": 1,
            }
        )
    return rows


# --- Build Dataset ---
num_fraud = int(NUM_TRANSACTIONS * FRAUD_RATE)  # ~135 fraud rows
num_normal = NUM_TRANSACTIONS - num_fraud

normal = generate_normal_transactions(num_normal)
fraud_bursts = generate_fraud_burst(n_bursts=8, txns_per_burst=6)  # 48
geo_mismatch = generate_geo_mismatch(n=6)  # 12
odd_hour = generate_odd_hour_high_value(n=8)  # 8

# Pad remaining fraud with random fraud-like transactions
remaining_fraud = num_fraud - len(fraud_bursts) - len(geo_mismatch) - len(odd_hour)
extra_fraud = generate_normal_transactions(remaining_fraud)
for r in extra_fraud:
    r["is_fraud"] = 1
    r["amount"] = round(np.random.uniform(10000, 60000), 2)

all_rows = normal + fraud_bursts + geo_mismatch + odd_hour + extra_fraud
df = pd.DataFrame(all_rows)

# Sort by timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Re-generate transaction IDs to be sequential
df["transaction_id"] = [f"TXN{i+1:06d}" for i in range(len(df))]

# Save
output_path = "transactions.csv"
df.to_csv(output_path, index=False)

# --- Summary ---
total = len(df)
fraud_count = df["is_fraud"].sum()
normal_count = total - fraud_count

print("=" * 50)
print("  SYNTHETIC FRAUD DATASET - SUMMARY")
print("=" * 50)
print(f"  Total transactions : {total}")
print(f"  Normal (is_fraud=0): {normal_count} ({normal_count/total*100:.1f}%)")
print(f"  Fraud   (is_fraud=1): {int(fraud_count)} ({fraud_count/total*100:.1f}%)")
print("-" * 50)
print(f"  Date range         : {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Unique cards       : {df['card_id'].nunique()}")
print(f"  Unique devices     : {df['device_id'].nunique()}")
print(f"  Unique merchants   : {df['merchant_id'].nunique()}")
print(f"  Amount range       : {df['amount'].min():.2f} - {df['amount'].max():.2f}")
print("-" * 50)
print("  Fraud patterns injected:")
print(f"    - Rapid bursts (same device/card): {len(fraud_bursts)} rows")
print(f"    - Geo-mismatch (impossible travel): {len(geo_mismatch)} rows")
print(f"    - Odd-hour high-value: {len(odd_hour)} rows")
print(f"    - Random high-value fill: {remaining_fraud} rows")
print("=" * 50)
print(f"\nSaved to: {output_path}")
