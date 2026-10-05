from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained ML model
model = joblib.load("linear_intelligence_file.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    hp = float(request.form["hp"])
    vol = float(request.form["vol"])
    sp = float(request.form["sp"])
    wt = float(request.form["wt"])

    # ML model prediction
    prediction = model.predict([[hp, vol, sp, wt]])

    mpg = round(float(prediction[0]), 2)

    return render_template(
        "index.html",
        prediction=mpg,
        hp=hp,
        vol=vol,
        sp=sp,
        wt=wt
    )


if __name__ == "__main__":
    app.run(debug=True)