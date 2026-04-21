# 🚗 Car Price Prediction System

## 📌 Overview
This project is a full-stack Machine Learning application that predicts the selling price of used cars based on multiple features such as fuel type, years of service, kilometers driven, and more.

The system leverages an optimized XGBoost regression model and provides predictions through both a REST API and an interactive Streamlit web interface.

---

## 🚀 Features
- 🔍 Predict car selling price with high accuracy (~94% R² score)
- ⚙️ Feature engineering (car age calculation)
- 🤖 Machine Learning using XGBoost with hyperparameter tuning
- 🌐 Flask API for backend prediction
- 🖥️ Streamlit UI for interactive user experience
- 📦 Model serialization using Pickle
- ☁️ Deployment-ready architecture

---

## 📊 Input Features
- Present Price
- Kilometers Driven
- Number of Previous Owners
- Car Age
- Fuel Type (Petrol/Diesel)
- Seller Type (Dealer/Individual)
- Transmission (Manual/Automatic)

---

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Flask
- Streamlit

---

## ⚙️ Project Structure
car-price-prediction/
│── train.py # Model training script
│── app.py # Flask API
│── streamlit_app.py # UI frontend
│── car.csv # Dataset
│── car_price_model.pkl # Trained model
│── requirements.txt
│── README.md


---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/car-price-prediction.git
cd car-price-prediction

2. Install dependencies
```bash
pip install -r requirements.txt

3. Run Streamlit UI
```bash
streamlit run streamlit_app.py