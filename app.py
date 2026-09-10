from flask import Flask, render_template, request
from LinearRegression import (
    calculateSales,
    buildRegressionPlot,
    N_RECORDS,
    R2_SCORE,
    SLOPE,
    INTERCEPT,
)
from LogisticRegression import (
    classifyStudent as classifyStudentLogistic,
    buildDataPlot as buildLogisticDataPlot,
    buildConfusionMatrixPlot as buildLogisticConfusionMatrixPlot,
    N_RECORDS as LOGISTIC_N_RECORDS,
    N_TRAIN as LOGISTIC_N_TRAIN,
    N_TEST as LOGISTIC_N_TEST,
    ACCURACY as LOGISTIC_ACCURACY,
    PRECISION as LOGISTIC_PRECISION,
    RECALL as LOGISTIC_RECALL,
    F1_SCORE as LOGISTIC_F1,
    TP as LOGISTIC_TP,
    TN as LOGISTIC_TN,
    FP as LOGISTIC_FP,
    FN as LOGISTIC_FN,
)
from RandomForest import (
    classifyStudent as classifyStudentRF,
    buildDataPlot as buildRFDataPlot,
    buildConfusionMatrixPlot as buildRFConfusionMatrixPlot,
    N_RECORDS as RF_N_RECORDS,
    N_TRAIN as RF_N_TRAIN,
    N_TEST as RF_N_TEST,
    ACCURACY as RF_ACCURACY,
    PRECISION as RF_PRECISION,
    RECALL as RF_RECALL,
    F1_SCORE as RF_F1,
    TP as RF_TP,
    TN as RF_TN,
    FP as RF_FP,
    FN as RF_FN,
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
        slope=SLOPE,
        intercept=INTERCEPT,
    )


@app.route("/supervised/logistic-regression/concepts")
def logistic_concepts():
    return render_template("logistic_concepts.html")


@app.route("/supervised/logistic-regression/application", methods=["GET", "POST"])
def logistic_application():
    result_class = None
    result_prob = None
    error = None
    input_value = ""

    if request.method == "POST":
        input_value = request.form.get("Hours", "").strip()

        if input_value == "":
            error = "Please enter a value for study hours."
        else:
            try:
                hours = float(input_value)
                result_class, result_prob = classifyStudentLogistic(hours)
            except ValueError:
                error = "The value entered must be numeric (e.g. 12.5)."

    plot_image = buildLogisticDataPlot()

    return render_template(
        "logistic_application.html",
        result_class=result_class,
        result_prob=result_prob,
        error=error,
        input_value=input_value,
        plot_image=plot_image,
        n_records=LOGISTIC_N_RECORDS,
        n_train=LOGISTIC_N_TRAIN,
        n_test=LOGISTIC_N_TEST,
    )


@app.route("/supervised/logistic-regression/evaluation-metrics")
def logistic_metrics():
    cm_image = buildLogisticConfusionMatrixPlot()
    return render_template(
        "logistic_metrics.html",
        cm_image=cm_image,
        n_test=LOGISTIC_N_TEST,
        accuracy=LOGISTIC_ACCURACY,
        precision=LOGISTIC_PRECISION,
        recall=LOGISTIC_RECALL,
        f1=LOGISTIC_F1,
        tp=LOGISTIC_TP,
        tn=LOGISTIC_TN,
        fp=LOGISTIC_FP,
        fn=LOGISTIC_FN,
    )


@app.route("/supervised/random-forest/concepts")
def rf_concepts():
    return render_template("rf_concepts.html")


@app.route("/supervised/random-forest/application", methods=["GET", "POST"])
def rf_application():
    result_class = None
    result_prob = None
    error = None
    input_hours = ""
    input_attendance = ""
    input_previous_score = ""

    if request.method == "POST":
        input_hours = request.form.get("Hours", "").strip()
        input_attendance = request.form.get("Attendance", "").strip()
        input_previous_score = request.form.get("PreviousScore", "").strip()

        if not input_hours or not input_attendance or not input_previous_score:
            error = "Please fill in all three fields."
        else:
            try:
                hours = float(input_hours)
                attendance = float(input_attendance)
                previous_score = float(input_previous_score)
                result_class, result_prob = classifyStudentRF(hours, attendance, previous_score)
            except ValueError:
                error = "All values must be numeric."

    plot_image = buildRFDataPlot()

    return render_template(
        "rf_application.html",
        result_class=result_class,
        result_prob=result_prob,
        error=error,
        input_hours=input_hours,
        input_attendance=input_attendance,
        input_previous_score=input_previous_score,
        plot_image=plot_image,
        n_records=RF_N_RECORDS,
        n_train=RF_N_TRAIN,
        n_test=RF_N_TEST,
    )


@app.route("/supervised/random-forest/evaluation-metrics")
def rf_metrics():
    cm_image = buildRFConfusionMatrixPlot()
    return render_template(
        "rf_metrics.html",
        cm_image=cm_image,
        n_test=RF_N_TEST,
        accuracy=RF_ACCURACY,
        precision=RF_PRECISION,
        recall=RF_RECALL,
        f1=RF_F1,
        tp=RF_TP,
        tn=RF_TN,
        fp=RF_FP,
        fn=RF_FN,
    )


if __name__ == "__main__":
    app.run(debug=True)