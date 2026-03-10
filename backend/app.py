# ==========================
# Imports
# ==========================
import os
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import pytz

app = Flask(__name__)

# ==========================
# Load Model Files
# ==========================
model = joblib.load("accident_risk_model.pkl")  # <-- FIXED filename
model_columns = joblib.load("model_columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ==========================
# Weather API
# ==========================
API_KEY = os.environ.get("API_KEY")  # <-- Use environment variable

# ==========================
# Fetch Weather
# ==========================
def get_weather(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        weather = data["weather"][0]["main"]
        weather = data["weather"][0]["main"]
        temperature = data["main"]["temp"]

        weather_map = {
            "Clear": 1,
            "Clouds": 2,
            "Rain": 3,
            "Drizzle": 3,
            "Thunderstorm": 4,
            "Fog": 5,
            "Mist": 5,
            "Haze": 5
        }

        return weather_map.get(weather, 1), f"{temperature}°C | {weather}"

    except:
        return 1

# ==========================
# Get Road Features (Overpass API)
# ==========================
def get_road_data(lat, lon):
    try:
        url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        way(around:50,{lat},{lon})["highway"];
        out tags;
        """
        response = requests.get(url, params={"data": query})
        data = response.json()

        if len(data["elements"]) == 0:
            return "residential", 40

        tags = data["elements"][0]["tags"]
        highway = tags.get("highway", "residential")
        speed = tags.get("maxspeed", "40")

        if speed.isdigit():
            speed = int(speed)
        else:
            speed = 40

        return highway, speed

    except:
        return "residential", 40

# ==========================
# Count Nearby Intersections
# ==========================
def get_nearby_clusters(lat, lon):
    try:
        url = "https://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        node(around:200,{lat},{lon})["highway"="traffic_signals"];
        out;
        """
        response = requests.get(url, params={"data": query})
        data = response.json()
        return len(data["elements"])

    except:
        return 1

# ==========================
# Convert Road Type
# ==========================
def map_road_type(highway):
    road_map = {
        "motorway": 3,
        "trunk": 3,
        "primary": 2,
        "secondary": 2,
        "tertiary": 2,
        "residential": 1,
        "service": 1
    }
    return road_map.get(highway, 1)

# ==========================
# Generate Model Features
# ==========================
def generate_features(lat, lon):
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    hour = current_time.hour
    if 5 <= hour < 12:
        time_of_day = "Morning"
    elif 12 <= hour < 17:
        time_of_day = "Afternoon"
    elif 17 <= hour < 21:
        time_of_day = "Evening"
    else:
        time_of_day = "Night"
    
    hour = now.hour
    is_night = 1 if (hour >= 20 or hour <= 5) else 0
    weather, weather_text = get_weather(lat, lon)

    # Fetch road data
    highway, speed_limit = get_road_data(lat, lon)
    road_type = map_road_type(highway)

    # Urban or rural approximation
    urban_rural = 1 if highway in ["residential", "tertiary"] else 0

    # Nearby intersections
    nearby_cluster = get_nearby_clusters(lat, lon)

    # Estimated Traffic Features
    heavy_ratio_map = {
        "motorway": 0.4,
        "primary": 0.3,
        "secondary": 0.2,
        "residential": 0.1
    }
    heavy_ratio = heavy_ratio_map.get(highway, 0.1)
    motorcycle_ratio = 0.5 if urban_rural == 1 else 0.3
    pedestrian_ratio = 0.3 if urban_rural == 1 else 0.1
    road_surface = 1
    
    traffic_score = heavy_ratio + motorcycle_ratio + pedestrian_ratio
    if traffic_score >= 0.9:
        traffic_level = "High"
    elif traffic_score >= 0.6:
        traffic_level = "Medium"
    else:
        traffic_level = "Low"
    reason = []
    if pedestrian_ratio >= 0.3:
        reason.append("more pedestrians")
    if motorcycle_ratio >= 0.5:
        reason.append("many motorcycles")
    if heavy_ratio >= 0.3:
        reason.append("heavy vehicles present")
    traffic_reason = ", ".join(reason) if reason else "normal traffic mix"
    # Derived Features
    lat_lon_interaction = lat * lon
    lat_squared = lat ** 2
    lon_squared = lon ** 2
    high_speed = 1 if speed_limit >= 60 else 0
    speed_urban = speed_limit * urban_rural
    speed_night = speed_limit * is_night

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
    return df, weather_text, highway, speed_limit, hour, nearby_cluster, urban_rural, traffic_level, traffic_reason
# ==========================
# Home Page
# ==========================
@app.route("/")
def home():
    return render_template("index.html")

# ==========================
# Prediction API
# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        lat = float(data["latitude"])
        lon = float(data["longitude"])
        features, weather_text, highway, speed_limit, hour, nearby_cluster, urban_rural, traffic_level, traffic_reason = generate_features(lat, lon)
        prediction = model.predict(features)
        risk_level = label_encoder.inverse_transform(prediction)[0]

        # Safety suggestions
        if risk_level == "High":
            solution = "High accident risk. Drive safely in this area. Reduce speed and increase traffic monitoring."
        elif risk_level == "Medium":
            solution = "Moderate risk area. Install warning signs and maintain road visibility."
        else:
            solution = "Road conditions appear relatively safe. Continue regular safety monitoring."

        # return jsonify({
        #     "risk_level": risk_level,   # <-- FIXED JSON key
        #     "weather": weather_text,
        #     "solution": solution,
        #     "factors": factors
        # })
        
        return jsonify({
    "risk_level": risk_level,
    "weather": weather_text,
    "solution": solution,
    "road_type": highway,
    "speed_limit": speed_limit,
    "time_of_day": time_of_day,
    "signals": nearby_cluster,
    "area": "Urban" if urban_rural == 1 else "Rural",
    "traffic": f"{traffic_level} ({traffic_reason})"
})

    except Exception as e:
        return jsonify({
            "risk_level": "Error",
            "solution": str(e)
        })

# ==========================
# Run Server
# ==========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
