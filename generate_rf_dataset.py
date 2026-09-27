import numpy as np
import pandas as pd

np.random.seed(11)

n_records = 480

study_hours = np.round(np.random.uniform(0, 25, n_records), 1)
attendance_pct = np.round(np.clip(np.random.normal(75, 15, n_records), 30, 100), 1)
previous_score = np.round(np.clip(np.random.normal(65, 18, n_records), 0, 100), 1)

z = (
    -7.5
    + 0.22 * study_hours
    + 0.05 * attendance_pct
    + 0.04 * previous_score
)
prob_pass = 1 / (1 + np.exp(-z))
passed = (np.random.uniform(0, 1, n_records) < prob_pass).astype(int)

df = pd.DataFrame({
    "study_hours_per_week": study_hours,
    "attendance_percentage": attendance_pct,
    "previous_exam_score": previous_score,
    "pass": passed
})

df.to_csv("data/student_performance_rf.csv", index=False)
print("Random Forest dataset:", len(df), "records")
print(df["pass"].value_counts())