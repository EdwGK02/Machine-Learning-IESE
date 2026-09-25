"""
Generates the dataset used by the SCIKIT-LEARN Clustering Application
(Part 2.3), related to the group's registered topic (grouping motorcycles
by technical characteristics). At least 1,000 records and 2+ numerical
variables are required; this generates 1,200 records with 6 numerical
variables:

    - engine_displacement_cc : engine displacement (cc)
    - engine_power_hp        : engine power (HP)
    - weight_kg              : curb weight (kg)
    - price_cop              : retail price (COP)
    - fuel_consumption_kmpl  : fuel economy (km/L)
    - max_speed_kmph         : top speed (km/h)

The data is synthetic but built from realistic, internally-consistent
relationships between the variables (bigger engines -> more power, more
weight, higher top speed, lower fuel economy, higher price), across four
plausible motorcycle segments, so K-Means has genuine structure to find:

    1. Urban / commuter          2. Naked / mixed-use
    3. Sport / high-performance  4. Touring / adventure

random_state is fixed so the dataset is reproducible for every teammate.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(7)

SEGMENTS = {
    "urban":    dict(n=350, cc=(100, 20, 90, 175),  power_k=0.09, weight_base=95,  weight_k=0.10, speed_base=70,  speed_k=0.06, kmpl_base=45, kmpl_k=-0.02, price_base=4_000_000,  price_k=16_000, noise=0.06),
    "naked":    dict(n=350, cc=(300, 60, 180, 520),  power_k=0.11, weight_base=130, weight_k=0.09, speed_base=110, speed_k=0.05, kmpl_base=34, kmpl_k=-0.015, price_base=9_000_000, price_k=24_000, noise=0.08),
    "sport":    dict(n=280, cc=(700, 150, 400, 1100), power_k=0.16, weight_base=150, weight_k=0.07, speed_base=150, speed_k=0.045, kmpl_base=22, kmpl_k=-0.010, price_base=22_000_000, price_k=36_000, noise=0.09),
    "touring":  dict(n=220, cc=(850, 180, 500, 1300), power_k=0.12, weight_base=210, weight_k=0.12, speed_base=130, speed_k=0.035, kmpl_base=20, kmpl_k=-0.008, price_base=30_000_000, price_k=32_000, noise=0.08),
}

rows = []
for seg_name, cfg in SEGMENTS.items():
    n = cfg["n"]
    mean_cc, std_cc, lo, hi = cfg["cc"]
    cc = np.random.normal(mean_cc, std_cc, n).clip(lo, hi)

    power = 8 + cfg["power_k"] * cc + np.random.normal(0, cfg["noise"] * 20, n)
    power = power.clip(6, None)

    weight = cfg["weight_base"] + cfg["weight_k"] * cc + np.random.normal(0, cfg["noise"] * 60, n)
    weight = weight.clip(75, None)

    speed = cfg["speed_base"] + cfg["speed_k"] * cc + np.random.normal(0, cfg["noise"] * 40, n)
    speed = speed.clip(60, 320)

    kmpl = cfg["kmpl_base"] + cfg["kmpl_k"] * cc + np.random.normal(0, cfg["noise"] * 15, n)
    kmpl = kmpl.clip(10, 60)

    price = cfg["price_base"] + cfg["price_k"] * cc + np.random.normal(0, cfg["noise"] * cfg["price_base"], n)
    price = price.clip(2_500_000, None)

    for i in range(n):
        rows.append({
            "engine_displacement_cc": round(float(cc[i]), 0),
            "engine_power_hp": round(float(power[i]), 1),
            "weight_kg": round(float(weight[i]), 1),
            "price_cop": round(float(price[i]), -3),
            "fuel_consumption_kmpl": round(float(kmpl[i]), 1),
            "max_speed_kmph": round(float(speed[i]), 0),
            "_true_segment": seg_name,  # kept only for our own sanity-check, dropped below
        })

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=7).reset_index(drop=True)  # shuffle
df.insert(0, "motorcycle_id", [f"MC{i+1:04d}" for i in range(len(df))])

true_segment = df.pop("_true_segment")  # keep separately, not part of the numerical features shown to K-Means

out_path = os.path.join(os.path.dirname(__file__), "data", "motorcycles_clustering.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)

print(f"Saved {len(df)} records to {out_path}")
print(df.describe())
print("\nSegment sizes (ground truth, for our reference only):")
print(true_segment.value_counts())
