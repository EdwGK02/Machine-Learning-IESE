import numpy as np
import pandas as pd

np.random.seed(42)

n_records = 550

temperature = np.random.uniform(-2, 40, n_records)

intercept = 15
slope = 5.8
noise = np.random.normal(0, 18, n_records)

sales = intercept + slope * temperature + noise
sales = np.clip(sales, 0, None)
sales = np.round(sales).astype(int)
temperature = np.round(temperature, 1)

df = pd.DataFrame({
    "temperature_celsius": temperature,
    "ice_cream_units_sold": sales
})

df.to_csv("data/ice_cream_sales.csv", index=False)
print(f"Dataset generated with {len(df)} records.")