import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import timedelta

fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

NUM_TRANSACTIONS = 1800  # Increased to accommodate hard examples
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


# ─────────────────────────────────────────────────────────────
# HARD NEGATIVES: Normal txns that trigger weak fraud signals
# These make the model's job harder and the AUC more realistic
# ─────────────────────────────────────────────────────────────
def generate_hard_negatives(n=150):
    """
    Genuine transactions that LOOK slightly suspicious:
    - Moderately high amounts (not extreme)
    - Some at slightly odd hours (6-7 AM, 11 PM-midnight)
    - Some with 2-3 txns in an hour (not 5+)
    - Some with moderate geo distance (200-400km, not 500+)
    All are is_fraud=0 but will confuse the model.
    """
    rows = []
    for _ in range(n):
        variant = random.choice(["high_amount", "odd_hour", "quick_pair", "moderate_travel", "mixed"])

        city = random.choice(LOC_NAMES)
        lat, lon = jitter_coords(LOCATIONS[city])
        card = random.choice(CARDS)
        device = random.choice(DEVICES)

        if variant == "high_amount":
            # Legitimate big purchase: electronics, furniture, etc.
            hour = random.choice(range(10, 22))  # normal hours
            ts = fake.date_time_between(start_date="-30d", end_date="now").replace(
                hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
            )
            rows.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(8000, 18000), 2),
                "timestamp": ts, "card_id": card, "device_id": device,
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat},{lon}", "is_fraud": 0,
            })

        elif variant == "odd_hour":
            # Late night online purchase (legitimate insomniac shopper)
            hour = random.choice([6, 7, 23, 0])
            ts = fake.date_time_between(start_date="-30d", end_date="now").replace(
                hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
            )
            rows.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(2000, 12000), 2),
                "timestamp": ts, "card_id": card, "device_id": device,
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat},{lon}", "is_fraud": 0,
            })

        elif variant == "quick_pair":
            # Two purchases within 15 min (buying coffee + gas, normal)
            base_time = fake.date_time_between(start_date="-30d", end_date="now")
            for _ in range(random.randint(2, 3)):
                ts = base_time + timedelta(minutes=random.randint(5, 20))
                rows.append({
                    "transaction_id": f"TXN{random.randint(100000, 999999)}",
                    "amount": round(np.random.uniform(500, 4000), 2),
                    "timestamp": ts, "card_id": card, "device_id": device,
                    "ip_address": random.choice(IP_POOL),
                    "merchant_id": random.choice(MERCHANTS),
                    "geo_location": f"{lat},{lon}", "is_fraud": 0,
                })

        elif variant == "moderate_travel":
            # Same card used 300km apart in 2 hours (train/highway travel)
            city2 = random.choice(LOC_NAMES)
            lat2, lon2 = jitter_coords(LOCATIONS[city2])
            base_time = fake.date_time_between(start_date="-30d", end_date="now")
            rows.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(1000, 6000), 2),
                "timestamp": base_time, "card_id": card, "device_id": device,
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat},{lon}", "is_fraud": 0,
            })
            rows.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(1000, 6000), 2),
                "timestamp": base_time + timedelta(hours=random.randint(1, 3)),
                "card_id": card, "device_id": device,
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat2},{lon2}", "is_fraud": 0,
            })

        elif variant == "mixed":
            # Slightly elevated amount at slightly odd hour
            hour = random.choice([0, 6, 7, 22, 23])
            ts = fake.date_time_between(start_date="-30d", end_date="now").replace(
                hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
            )
            rows.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(6000, 14000), 2),
                "timestamp": ts, "card_id": card, "device_id": device,
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat},{lon}", "is_fraud": 0,
            })

    return rows


# ─────────────────────────────────────────────────────────────
# AGGRESSIVE HARD NEGATIVES: Very fraud-like but genuine
# These MUST create overlap in the 0.3-0.7 score range
# ─────────────────────────────────────────────────────────────
def generate_aggressive_hard_negatives(n=80):
    """
    Normal transactions that closely mimic fraud patterns:
    - High amounts at odd-ish hours
    - 3-4 txns in an hour (just below burst threshold)
    - Same card used in 2 cities within 1 hour (travel, not fraud)
    """
    rows = []
    for _ in range(n):
        variant = random.choice(["near_burst", "near_geo", "high_odd", "combo"])
        card = random.choice(CARDS)
        device = random.choice(DEVICES)

        if variant == "near_burst":
            # 4 txns in 30 min (rule needs 5 in 5 min to fire)
            city = random.choice(LOC_NAMES)
            lat, lon = jitter_coords(LOCATIONS[city])
            base_time = fake.date_time_between(start_date="-30d", end_date="now")
            for _ in range(4):
                t = base_time + timedelta(minutes=random.randint(3, 10))
                rows.append({
                    "transaction_id": f"TXN{random.randint(100000, 999999)}",
                    "amount": round(np.random.uniform(3000, 10000), 2),
                    "timestamp": t, "card_id": card, "device_id": device,
                    "ip_address": random.choice(IP_POOL),
                    "merchant_id": random.choice(MERCHANTS),
                    "geo_location": f"{lat},{lon}", "is_fraud": 0,
                })

        elif variant == "near_geo":
            # Same card, 2 cities, 45 min apart (fast train)
            city1, city2 = random.sample(LOC_NAMES, 2)
            lat1, lon1 = jitter_coords(LOCATIONS[city1])
            lat2, lon2 = jitter_coords(LOCATIONS[city2])
            base_time = fake.date_time_between(start_date="-30d", end_date="now")
            rows.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(2000, 8000), 2),
                "timestamp": base_time, "card_id": card, "device_id": device,
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat1},{lon1}", "is_fraud": 0,
            })
            rows.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(2000, 8000), 2),
                "timestamp": base_time + timedelta(minutes=random.randint(40, 90)),
                "card_id": card, "device_id": device,
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat2},{lon2}", "is_fraud": 0,
            })

        elif variant == "high_odd":
            # High amount at 1-5 AM (online international purchase)
            city = random.choice(LOC_NAMES)
            lat, lon = jitter_coords(LOCATIONS[city])
            hour = random.choice([1, 2, 3, 4])
            ts = fake.date_time_between(start_date="-30d", end_date="now").replace(
                hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
            )
            rows.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(12000, 25000), 2),
                "timestamp": ts, "card_id": card, "device_id": device,
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat},{lon}", "is_fraud": 0,
            })

        elif variant == "combo":
            # Multiple weak signals combined
            city = random.choice(LOC_NAMES)
            lat, lon = jitter_coords(LOCATIONS[city])
            hour = random.choice([0, 1, 5, 6, 23])
            ts = fake.date_time_between(start_date="-30d", end_date="now").replace(
                hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
            )
            card2 = random.choice(CARDS)
            rows.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(8000, 18000), 2),
                "timestamp": ts, "card_id": card, "device_id": device,
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat},{lon}", "is_fraud": 0,
            })

    return rows


# ─────────────────────────────────────────────────────────────
# HARD POSITIVES: Subtle fraud that avoids obvious patterns
# These score 0.5-0.8 instead of 0.95+
# ─────────────────────────────────────────────────────────────
def generate_hard_positives(n=40):
    """
    Fraud that is deliberately subtle:
    - Moderate amounts (not extreme)
    - Spread across normal hours
    - No rapid bursts (separate devices/cards)
    - No impossible travel
    - Only 1 weak signal each (e.g., slightly above card avg)
    """
    rows = []
    for _ in range(n):
        variant = random.choice(["sleeper_card", "small_theft", "velocity_lite", "new_device", "muled"])

        city = random.choice(LOC_NAMES)
        lat, lon = jitter_coords(LOCATIONS[city])
        hour = random.choice(range(8, 22))  # normal business hours
        ts = fake.date_time_between(start_date="-30d", end_date="now").replace(
            hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
        )

        if variant == "sleeper_card":
            # Stolen card used for moderate purchase after dormancy
            rows.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(3000, 9000), 2),
                "timestamp": ts, "card_id": random.choice(CARDS),
                "device_id": random.choice(DEVICES),
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat},{lon}", "is_fraud": 1,
            })

        elif variant == "small_theft":
            # Multiple small fraudulent charges to avoid detection
            card = random.choice(CARDS)
            for _ in range(random.randint(2, 3)):
                t = ts + timedelta(hours=random.randint(1, 6))
                rows.append({
                    "transaction_id": f"TXN{random.randint(100000, 999999)}",
                    "amount": round(np.random.uniform(1500, 5000), 2),
                    "timestamp": t, "card_id": card,
                    "device_id": random.choice(DEVICES),
                    "ip_address": random.choice(IP_POOL),
                    "merchant_id": random.choice(MERCHANTS),
                    "geo_location": f"{lat},{lon}", "is_fraud": 1,
                })

        elif variant == "velocity_lite":
            # 3 txns in an hour (not 5+, so rule doesn't fire)
            card = random.choice(CARDS)
            device = random.choice(DEVICES)
            for _ in range(3):
                t = ts + timedelta(minutes=random.randint(10, 30))
                rows.append({
                    "transaction_id": f"TXN{random.randint(100000, 999999)}",
                    "amount": round(np.random.uniform(2000, 7000), 2),
                    "timestamp": t, "card_id": card, "device_id": device,
                    "ip_address": random.choice(IP_POOL),
                    "merchant_id": random.choice(MERCHANTS),
                    "geo_location": f"{lat},{lon}", "is_fraud": 1,
                })

        elif variant == "new_device":
            # Fraud using a new device but normal amount
            rows.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(4000, 10000), 2),
                "timestamp": ts, "card_id": random.choice(CARDS),
                "device_id": f"D{random.randint(90000, 99999)}",  # rare device
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat},{lon}", "is_fraud": 1,
            })

        elif variant == "muled":
            # Money mule: moderate amount, slightly unusual merchant
            rows.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(5000, 12000), 2),
                "timestamp": ts, "card_id": random.choice(CARDS),
                "device_id": random.choice(DEVICES),
                "ip_address": random.choice(IP_POOL),
                "merchant_id": "M0049",  # high-risk merchant
                "geo_location": f"{lat},{lon}", "is_fraud": 1,
            })

    return rows


# ─────────────────────────────────────────────────────────────
# BORDERLINE FRAUD: Transactions that are 50/50
# These MUST populate the 0.3-0.7 score range
# ─────────────────────────────────────────────────────────────
def generate_borderline_fraud(n=30):
    """
    Fraud that almost looks legitimate:
    - Normal amounts
    - Normal hours
    - But card is slightly over its spending limit
    - Or 2 txns within 45 min (could be legitimate)
    """
    rows = []
    for _ in range(n):
        variant = random.choice(["over_limit", "paired_subtle", "merchant_switch"])
        city = random.choice(LOC_NAMES)
        lat, lon = jitter_coords(LOCATIONS[city])

        if variant == "over_limit":
            hour = random.choice(range(9, 21))
            ts = fake.date_time_between(start_date="-30d", end_date="now").replace(
                hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
            )
            rows.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(4000, 8000), 2),
                "timestamp": ts, "card_id": random.choice(CARDS),
                "device_id": random.choice(DEVICES),
                "ip_address": random.choice(IP_POOL),
                "merchant_id": random.choice(MERCHANTS),
                "geo_location": f"{lat},{lon}", "is_fraud": 1,
            })

        elif variant == "paired_subtle":
            card = random.choice(CARDS)
            base_time = fake.date_time_between(start_date="-30d", end_date="now")
            for _ in range(2):
                t = base_time + timedelta(minutes=random.randint(20, 45))
                rows.append({
                    "transaction_id": f"TXN{random.randint(100000, 999999)}",
                    "amount": round(np.random.uniform(2000, 6000), 2),
                    "timestamp": t, "card_id": card,
                    "device_id": random.choice(DEVICES),
                    "ip_address": random.choice(IP_POOL),
                    "merchant_id": random.choice(MERCHANTS),
                    "geo_location": f"{lat},{lon}", "is_fraud": 1,
                })

        elif variant == "merchant_switch":
            card = random.choice(CARDS)
            hour = random.choice(range(10, 22))
            ts = fake.date_time_between(start_date="-30d", end_date="now").replace(
                hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59)
            )
            rows.append({
                "transaction_id": f"TXN{random.randint(100000, 999999)}",
                "amount": round(np.random.uniform(3000, 7000), 2),
                "timestamp": ts, "card_id": card,
                "device_id": random.choice(DEVICES),
                "ip_address": random.choice(IP_POOL),
                "merchant_id": "M0049",
                "geo_location": f"{lat},{lon}", "is_fraud": 1,
            })

    return rows


# --- Build Dataset ---
num_fraud = int(NUM_TRANSACTIONS * FRAUD_RATE)  # ~162 fraud rows
num_normal = NUM_TRANSACTIONS - num_fraud

normal = generate_normal_transactions(num_normal - 150 - 80)  # leave room for hard examples
fraud_bursts = generate_fraud_burst(n_bursts=8, txns_per_burst=6)  # 48
geo_mismatch = generate_geo_mismatch(n=6)  # 12
odd_hour = generate_odd_hour_high_value(n=8)  # 8

# Hard examples — the key to realistic AUC
hard_negatives = generate_hard_negatives(n=150)  # normal but trigger weak signals
aggressive_negatives = generate_aggressive_hard_negatives(n=80)  # very fraud-like normal
hard_positives = generate_hard_positives(n=40)   # fraud but subtle
borderline_fraud = generate_borderline_fraud(n=30)  # 50/50 fraud

# Pad remaining fraud with random fraud-like transactions
obvious_fraud = len(fraud_bursts) + len(geo_mismatch) + len(odd_hour) + len(hard_positives) + len(borderline_fraud)
remaining_fraud = max(0, num_fraud - obvious_fraud)
extra_fraud = generate_normal_transactions(remaining_fraud)
for r in extra_fraud:
    r["is_fraud"] = 1
    r["amount"] = round(np.random.uniform(10000, 60000), 2)

all_rows = normal + hard_negatives + aggressive_negatives + fraud_bursts + geo_mismatch + odd_hour + hard_positives + borderline_fraud + extra_fraud
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
print(f"    - Subtle fraud (hard positives): {len(hard_positives)} rows")
print(f"    - Borderline fraud: {len(borderline_fraud)} rows")
print(f"    - Random high-value fill: {remaining_fraud} rows")
print(f"    - Hard negatives (ambiguous normal): {len(hard_negatives)} rows")
print(f"    - Aggressive hard negatives: {len(aggressive_negatives)} rows")
print("=" * 50)
print(f"\nSaved to: {output_path}")
