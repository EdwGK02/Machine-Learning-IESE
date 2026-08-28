from flask import Flask, render_template

app = Flask(__name__, template_folder="template")

@app.route("/")
def home():
    return "hello word"

@app.route("/template")
def template():
    return render_template("index.html")