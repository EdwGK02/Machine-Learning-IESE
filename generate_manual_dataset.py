"""
Generates the 100-record dataset used for the MANUAL K-Means simulation
(Part 1 of the assignment). Two numerical variables are used, both directly
tied to the group's topic (grouping motorcycles by technical characteristics):

    - engine_displacement_cc : engine displacement, in cubic centimeters (cc)
    - price_cop              : retail price, in Colombian pesos (COP)

The data is synthetically generated (not scraped) but modeled on realistic
ranges for three well-known motorcycle segments in the Colombian market, so
that three natural groups exist for the manual exercise:

    1. Urban / commuter motorcycles   (~100-200 cc,  low price)
    2. Mid-size / all-purpose motorcycles (~250-500 cc, mid price)
    3. High-performance / touring motorcycles (~600-1300 cc, high price)

Running this script regenerates data/motorcycles_manual.csv deterministically
(random_state fixed) so every teammate gets the exact same 100 records.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

N_PER_GROUP = [34, 33, 33]  # 100 records total

# --- Group 1: Urban / commuter motorcycles ---
cc_1 = np.random.normal(150, 25, N_PER_GROUP[0]).clip(100, 200)
price_1 = 4_500_000 + (cc_1 - 100) * 15_000 + np.random.normal(0, 400_000, N_PER_GROUP[0])

# --- Group 2: Mid-size / all-purpose motorcycles ---
cc_2 = np.random.normal(375, 55, N_PER_GROUP[1]).clip(250, 500)
price_2 = 11_000_000 + (cc_2 - 250) * 28_000 + np.random.normal(0, 900_000, N_PER_GROUP[1])

# --- Group 3: High-performance / touring motorcycles ---
cc_3 = np.random.normal(900, 180, N_PER_GROUP[2]).clip(600, 1300)
price_3 = 28_000_000 + (cc_3 - 600) * 42_000 + np.random.normal(0, 2_500_000, N_PER_GROUP[2])

cc = np.concatenate([cc_1, cc_2, cc_3])
price = np.concatenate([price_1, price_2, price_3])
price = np.clip(price, 3_000_000, None)

order = np.random.permutation(len(cc))
cc = cc[order]
price = price[order]

df = pd.DataFrame({
    "motorcycle_id": [f"M{i+1:03d}" for i in range(len(cc))],
    "engine_displacement_cc": np.round(cc, 0).astype(int),
    "price_cop": np.round(price, -3).astype(int),  # round to nearest 1,000 COP
})

out_path = os.path.join(os.path.dirname(__file__), "data", "motorcycles_manual.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
df.to_csv(out_path, index=False)

print(f"Saved {len(df)} records to {out_path}")
print(df.describe())
