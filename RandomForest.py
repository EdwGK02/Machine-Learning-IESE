import os
import io
import base64

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "student_performance_rf.csv")

df = pd.read_csv(DATA_PATH)

FEATURES = ["study_hours_per_week", "attendance_percentage", "previous_exam_score"]

x = df[FEATURES]
y = df["pass"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

N_RECORDS = len(df)
N_TRAIN = len(x_train)
N_TEST = len(x_test)

ACCURACY = round(accuracy_score(y_test, y_pred), 4)
PRECISION = round(precision_score(y_test, y_pred), 4)
RECALL = round(recall_score(y_test, y_pred), 4)
F1_SCORE = round(f1_score(y_test, y_pred), 4)

CM = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = CM.ravel()


def classifyStudent(hours, attendance, previous_score):
    input_df = pd.DataFrame({
        "study_hours_per_week": [hours],
        "attendance_percentage": [attendance],
        "previous_exam_score": [previous_score],
    })
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    return int(prediction), round(float(probability), 4)


def buildDataPlot():
    fig, ax = plt.subplots(figsize=(8, 5))

    failed = df[df["pass"] == 0]
    passed = df[df["pass"] == 1]

    ax.scatter(failed["study_hours_per_week"], failed["previous_exam_score"],
               alpha=0.55, color="#ff6b9d", label="Fail (0)", s=22)
    ax.scatter(passed["study_hours_per_week"], passed["previous_exam_score"],
               alpha=0.55, color="#4ecdc4", label="Pass (1)", s=22)

    ax.set_title("Random Forest: Study Hours vs. Previous Score, by Class")
    ax.set_xlabel("Study Hours per Week")
    ax.set_ylabel("Previous Exam Score")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def buildConfusionMatrixPlot():
    fig, ax = plt.subplots(figsize=(5, 4.5))

    ax.imshow(CM, cmap="RdPu")

    labels = ["Fail (0)", "Pass (1)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(CM[i, j]), ha="center", va="center",
                     color="white" if CM[i, j] > CM.max() / 2 else "black",
                     fontsize=14, fontweight="bold")

    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")