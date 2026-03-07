from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os # Added for Render environment variables

app = Flask(__name__)

# ==========================
# Load Model Files (FIXED NAMES)
# ==========================
# I've updated these to match your repo's filenames
model = joblib.load("accident_risk_model.pkl") 
model_columns = joblib.load("model_columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ==========================
# Weather API (RENDER SECURE)
# ==========================
# This pulls from your Render "Environment" settings
API_KEY = os.environ.get("API_KEY")

# ==========================
# Fetch Weather
# ==========================
def get_weather(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()
        weather = data["weather"][0]["main"]

        weather_map = {
            "Clear": 1, "Clouds": 2, "Rain": 3, "Drizzle": 3,
            "Thunderstorm": 4, "Fog": 5, "Mist": 5, "Haze": 5
        }
        return weather_map.get(weather, 1)
    except:
        return 1

# ==========================
# Get Road Features (Overpass API)
# ==========================
def get_road_data(lat, lon):
    try:
        url = "https://overpass-api.de/api/interpreter"
        query = f'[out:json];way(around:50,{lat},{lon})["highway"];out tags;'
        response = requests.get(url, params={"data": query}, timeout=10)
        data = response.json()

        if not data["elements"]:
            return "residential", 40

        tags = data["elements"][0]["tags"]
        highway = tags.get("highway", "residential")
        speed = tags.get("maxspeed", "40")

        # Clean speed string
        speed = int(''.join(filter(str.isdigit, speed))) if any(c.isdigit() for c in speed) else 40
        return highway, speed
    except:
        return "residential", 40

# ==========================
# Count Nearby Intersections
# ==========================
def get_nearby_clusters(lat, lon):
    try:
        url = "https://overpass-api.de/api/interpreter"
        query = f'[out:json];node(around:200,{lat},{lon})["highway"="traffic_signals"];out;'
        response = requests.get(url, params={"data": query}, timeout=10)
        data = response.json()
        return len(data["elements"])
    except:
        return 1

# ==========================
# Convert Road Type
# ==========================
def map_road_type(highway):
    road_map = {
        "motorway": 3, "trunk": 3, "primary": 2, 
        "secondary": 2, "tertiary": 2, "residential": 1, "service": 1
    }
    return road_map.get(highway, 1)

# ==========================
# Generate Model Features
# ==========================
def generate_features(lat, lon):
    now = datetime.now()
    hour = now.hour
    is_night = 1 if (hour >= 20 or hour <= 5) else 0

    weather = get_weather(lat, lon)
    highway, speed_limit = get_road_data(lat, lon)
    road_type = map_road_type(highway)

    urban_rural = 1 if highway in ["residential", "tertiary"] else 0
    nearby_cluster = get_nearby_clusters(lat, lon)

    # Her Heuristic Ratios (The 'Estimation' strategy)
    heavy_ratio_map = {"motorway": 0.4, "primary": 0.3, "secondary": 0.2, "residential": 0.1}
    heavy_ratio = heavy_ratio_map.get(highway, 0.1)

    if urban_rural == 1:
        motorcycle_ratio, pedestrian_ratio = 0.5, 0.3
    else:
        motorcycle_ratio, pedestrian_ratio = 0.3, 0.1

    road_surface = 1
    lat_lon_interaction = lat * lon
    lat_squared = lat ** 2
    lon_squared = lon ** 2

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
        "High_Speed": 1 if speed_limit >= 60 else 0,
        "Speed_Urban": speed_limit * urban_rural,
        "Speed_Night": speed_limit * is_night
    }

    df = pd.DataFrame([data])
    df = df.reindex(columns=model_columns, fill_value=0)
    return df

# ==========================
# Routes
# ==========================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        lat, lon = float(data["latitude"]), float(data["longitude"])
        features = generate_features(lat, lon)
        prediction = model.predict(features)
        risk_level = label_encoder.inverse_transform(prediction)[0]

        # Use the key "risk" to match your index.html/script.js logic
        solutions = {
            "High": "Install speed breakers, warning signs, and improve lighting.",
            "Medium": "Add caution boards and monitor traffic conditions.",
            "Low": "Road conditions appear relatively safe."
        }

        return jsonify({
            "risk": risk_level, # Changed to "risk" to match her UI logic
            "solution": solutions.get(risk_level, "")
        })
    except Exception as e:
        return jsonify({"risk": "Error", "solution": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
