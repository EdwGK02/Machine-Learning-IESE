import numpy as np
import pandas as pd

np.random.seed(7)

n_records = 480

study_hours = np.round(np.random.uniform(0, 25, n_records), 1)

z = -4.2 + 0.38 * study_hours
prob_pass = 1 / (1 + np.exp(-z))
passed = (np.random.uniform(0, 1, n_records) < prob_pass).astype(int)

df = pd.DataFrame({
    "study_hours_per_week": study_hours,
    "pass": passed
})

df.to_csv("data/student_performance_logistic.csv", index=False)
print("Logistic dataset:", len(df), "records")
print(df["pass"].value_counts())