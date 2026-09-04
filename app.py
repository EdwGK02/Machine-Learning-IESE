from flask import Flask, render_template, request
from LinearRegression import (
    calculateSales,
    buildRegressionPlot,
    N_RECORDS,
    R2_SCORE,
    MAE_SCORE,
    SLOPE,
    INTERCEPT,
)

app = Flask(__name__, template_folder="template")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/machine-learning/concepts")
def ml_concepts():
    return render_template("ml_concepts.html")


@app.route("/machine-learning/types")
def ml_types():
    return render_template("ml_types.html")


@app.route("/machine-learning/use-cases/<int:n>")
def use_case(n):
    if n not in (1, 2, 3, 4):
        return "Use case not found", 404
    return render_template(f"use_case{n}.html")


@app.route("/supervised/linear-regression/concepts")
def lr_concepts():
    return render_template("lr_concepts.html")


@app.route("/supervised/linear-regression/application", methods=["GET", "POST"])
def lr_application():
    result = None
    error = None
    input_value = ""

    if request.method == "POST":
        input_value = request.form.get("Temperature", "").strip()

        if input_value == "":
            error = "Please enter a temperature value."
        else:
            try:
                temperature = float(input_value)
                result = calculateSales(temperature)
            except ValueError:
                error = "The value entered must be numeric (e.g. 24.5)."

    plot_image = buildRegressionPlot()

    return render_template(
        "tempLinearRegression.html",
        result=result,
        error=error,
        input_value=input_value,
        plot_image=plot_image,
        n_records=N_RECORDS,
        r2_score=R2_SCORE,
        mae_score=MAE_SCORE,
        slope=SLOPE,
        intercept=INTERCEPT,
    )


if __name__ == "__main__":
    app.run(debug=True)