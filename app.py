# =============================
# app.py (RUN THIS AFTER TRAINING)
# =============================

from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open('car_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return "Car Price Prediction API Running"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)[0]
    return jsonify({'predicted_price': float(prediction)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)