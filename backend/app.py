from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import requests

app = Flask(__name__)

# Load model files
model = joblib.load("accident_risk_model.pkl")
model_columns = joblib.load("model_columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Weather API
import os

API_KEY = os.environ.get("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Set it on Render.")


# ----------------------------
# Fetch Weather Data
# ----------------------------
def get_weather(lat, lon):

    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}"

    try:
        response = requests.get(url, timeout=5).json()
        weather = response["weather"][0]["main"]
    except:
        weather = "Clear"

    # Convert weather to dataset codes (simplified)
    weather_map = {
    "Clear": 1,
    "Clouds": 2,
    "Rain": 3,
    "Drizzle": 3,
    "Thunderstorm": 3,
    "Mist": 4,
    "Fog": 4,
    "Haze": 4
    }

    return weather_map.get(weather, 1)


# ----------------------------
# Generate Other Features
# ----------------------------
def generate_features(lat, lon):

    now = datetime.now()

    hour = now.hour

    is_night = 1 if (hour >= 20 or hour <= 5) else 0

    speed_limit = 60
    road_type = 1
    urban_rural = 1
    road_surface = 1

    heavy_ratio = 0.2
    motorcycle_ratio = 0.1
    pedestrian_ratio = 0.05

    lat_lon_interaction = lat * lon
    lat_squared = lat ** 2
    lon_squared = lon ** 2

    nearby_cluster = 3

    high_speed = 1 if speed_limit >= 60 else 0
    speed_urban = speed_limit * urban_rural
    speed_night = speed_limit * is_night

    weather = get_weather(lat, lon)

    data = {
        "Speed_limit": speed_limit,
        "Urban_or_Rural_Area": urban_rural,
        "Road_Type": road_type,
        "Weather_Conditions": weather,
        "Light_Conditions": is_night,
        "Road_Surface_Conditions": road_surface,
        "Hour": hour,
        "latitude": lat,
        "longitude": lon,
        "Heavy_Vehicle_Ratio": heavy_ratio,
        "Motorcycle_Ratio": motorcycle_ratio,
        "Pedestrian_Ratio": pedestrian_ratio,
        "lat_lon_interaction": lat_lon_interaction,
        "lat_squared": lat_squared,
        "lon_squared": lon_squared,
        "Nearby_Cluster_Count": nearby_cluster,
        "Is_Night": is_night,
        "High_Speed": high_speed,
        "Speed_Urban": speed_urban,
        "Speed_Night": speed_night
    }

    df = pd.DataFrame([data])

    df = pd.get_dummies(df)

    df = df.reindex(columns=model_columns, fill_value=0)

    return df


# ----------------------------
# API ROUTE
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    lat = float(data["latitude"])
    lon = float(data["longitude"])

    features = generate_features(lat, lon)

    prediction = model.predict(features)

    risk = label_encoder.inverse_transform(prediction)[0]

    return jsonify({
        "risk_level": risk
    })


@app.route("/")
def home():
    
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
