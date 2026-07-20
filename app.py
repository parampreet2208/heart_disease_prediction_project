from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

pipeline = joblib.load("heart_pipeline.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {
        "age": int(request.form["age"]),
        "sex": request.form["sex"],
        "chest_pain_type": request.form["chest_pain_type"],
        "resting_blood_pressure": int(request.form["resting_blood_pressure"]),
        "cholestoral": int(request.form["cholestoral"]),
        "fasting_blood_sugar": request.form["fasting_blood_sugar"],
        "rest_ecg": request.form["rest_ecg"],
        "Max_heart_rate": int(request.form["Max_heart_rate"]),
        "exercise_induced_angina": request.form["exercise_induced_angina"],
        "oldpeak": float(request.form["oldpeak"]),
        "slope": request.form["slope"],
        "vessels_colored_by_flourosopy": request.form["vessels"],
        "thalassemia": request.form["thal"]
    }

    input_df = pd.DataFrame([data])

    prediction = pipeline.predict(input_df)

    if prediction[0] == 1:
        result = "❤️ Heart Disease Detected"
    else:
        result = "💚 No Heart Disease Detected"

    return render_template("index.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)