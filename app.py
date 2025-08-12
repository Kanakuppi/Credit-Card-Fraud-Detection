from flask import Flask, request, render_template
import pandas as pd
import joblib

LOGISTIC_MODEL_FILE = "logistic_model.pkl"
DECISION_MODEL_FILE = "decision_model.pkl"

# Load models
try:
    logistic_model = joblib.load(LOGISTIC_MODEL_FILE)
    decision_model = joblib.load(DECISION_MODEL_FILE)
except FileNotFoundError:
    print("Error: Model files not found. Please ensure 'logistic_model.pkl' and 'decision_model.pkl' are in the correct directory.")
    logistic_model = None
    decision_model = None

app = Flask(__name__)
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if logistic_model is None or decision_model is None:
        return render_template("index.html", prediction="Error: Models not loaded.")

    try:
        # Collect form data for all features
        data = {}
        data['Time'] = float(request.form['Time'])
        for i in range(1, 29):
            data[f'V{i}'] = float(request.form[f'V{i}'])
        data['Amount'] = float(request.form['Amount'])

        df = pd.DataFrame([data])

        model_type = request.form.get('model_type')

        if model_type == 'logistic':
            prediction = logistic_model.predict(df)[0]
        elif model_type == 'decision':
            prediction = decision_model.predict(df)[0]
        else:
            return render_template("index.html", prediction="Error: Invalid model type selected.")

        outcome = "Fraud" if prediction == 1 else "Legit"

        return render_template("index.html", prediction=f"The transaction is predicted as: {outcome}")
    except Exception as e:
        import logging
        logging.exception("Prediction Error:")
        return render_template("index.html", prediction=f"Error: Invalid input. Please check your values. Details: {e}")

if __name__ == "__main__":
    # To run in Colab, you need to use ngrok or a similar service
    # from google.colab.output import eval_js
    # print(eval_js('google.colab.kernel.proxyPort(5000)'))
    app.run(debug=True, host='0.0.0.0', port=5000)