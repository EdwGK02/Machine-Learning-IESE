import os
import io
import base64

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "ice_cream_sales.csv")

df = pd.read_csv(DATA_PATH)

x = df[["temperature_celsius"]]
y = df["ice_cream_units_sold"]

model = LinearRegression()
model.fit(x, y)

predictions_on_training_data = model.predict(x)
R2_SCORE = round(r2_score(y, predictions_on_training_data), 4)
MAE_SCORE = round(mean_absolute_error(y, predictions_on_training_data), 2)
SLOPE = round(float(model.coef_[0]), 4)
INTERCEPT = round(float(model.intercept_), 4)
N_RECORDS = len(df)


def calculateSales(temperature):
    input_df = pd.DataFrame({"temperature_celsius": [temperature]})
    result = model.predict(input_df)[0]
    return round(max(result, 0), 2)


def buildRegressionPlot():
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df["temperature_celsius"], df["ice_cream_units_sold"],
               alpha=0.5, color="#667eea", label="Historical data", s=18)

    x_line_values = np.linspace(df["temperature_celsius"].min(),
                                 df["temperature_celsius"].max(), 100)
    x_line = pd.DataFrame({"temperature_celsius": x_line_values})
    y_line = model.predict(x_line)
    ax.plot(x_line_values, y_line, color="#E91E63", linewidth=2.5, label="Regression line")

    ax.set_title("Ice Cream Sales vs. Temperature", fontsize=14, fontweight="bold")
    ax.set_xlabel("Temperature (°C)", fontsize=11)
    ax.set_ylabel("Ice Cream Units Sold", fontsize=11)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")