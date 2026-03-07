from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os

app = Flask(__name__)

# ----------------------------
# 1. Load Model Files (FIXED NAMES)
# ----------------------------
model = joblib.load("accident_risk_model.pkl")  # Fixed from blackspot_model
model_columns = joblib.load("model_columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")

API_KEY = os.environ.get("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not found. Please set it in your Render Environment Variables.")

# ----------------------------
# 2. Real-Time Data Helpers
# ----------------------------

def get_realtime_data(lat, lon):
    weather_code, is_night, road_surface = 1, 0, 1
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}"
    
    try:
        w_res = requests.get(weather_url, timeout=5).json()
        main_weather = w_res["weather"][0]["main"]
        now = datetime.now().timestamp()
        is_night = 1 if (now < w_res['sys']['sunrise'] or now > w_res['sys']['sunset']) else 0
        if main_weather in ["Rain", "Drizzle", "Thunderstorm", "Snow"]:
            road_surface = 2 
        weather_map = {"Clear": 1, "Clouds": 2, "Rain": 3, "Drizzle": 3, "Mist": 4, "Fog": 4}
        weather_code = weather_map.get(main_weather, 1)
    except:
        print("Weather API failed.")

    # Overpass API for Road Type & Signals
    speed_limit, urban_rural, road_type, signals = 40, 1, 1, 0
    overpass_url = "https://overpass-api.de/api/interpreter" # Changed to https
    overpass_query = f"""
    [out:json];
    (
      way(around:50, {lat}, {lon})["highway"];
      node(around:200, {lat}, {lon})["highway"="traffic_signals"];
    );
    out tags;
    """
    try:
        o_res = requests.get(overpass_url, params={'data': overpass_query}, timeout=10).json()
        elements = o_res.get('elements', [])
        
        # Distinguish between the Road (way) and Signals (node)
        ways = [e for e in elements if e['type'] == 'way']
        signals = len([e for e in elements if e['type'] == 'node'])
        
        if ways:
            tags = ways[0].get('tags', {})
            speed_val = tags.get("maxspeed", "40")
            speed_limit = int(''.join(filter(str.isdigit, speed_val))) if any(c.isdigit() for c in speed_val) else 40
            highway = tags.get("highway", "residential")
            urban_rural = 2 if highway in ["motorway", "trunk", "primary"] else 1
            # Road Type mapping (1: Local, 2: Main, 3: Highway)
            road_type = 3 if highway in ["motorway", "trunk"] else (2 if highway in ["primary", "secondary"] else 1)
    except:
        print("Overpass API failed.")

    return weather_code, is_night, road_surface, speed_limit, urban_rural, road_type, signals

# ----------------------------
# 3. Feature Engineering
# ----------------------------

def generate_features(lat, lon):
    weather, is_night, surface, speed, urban_rural, road_type, signals = get_realtime_data(lat, lon)
    
    now = datetime.now()
    hour = now.hour

    # The "Hybrid" Logic: Using your Colab-trained averages
    # But varying them slightly based on road type to get different results
    if road_type >= 2: # Main roads/Highways
        heavy_ratio = 0.728
        nearby_cluster = 32 + (signals * 2)
    else: # Residential
        heavy_ratio = 0.705
        nearby_cluster = 24 + signals

    data = {
        "Speed_limit": speed,
        "Urban_or_Rural_Area": urban_rural,
        "Road_Type": road_type,
        "Weather_Conditions": weather,
        "Light_Conditions": is_night,
        "Road_Surface_Conditions": surface,
        "Hour": hour,
        "latitude": lat,
        "longitude": lon,
        "Heavy_Vehicle_Ratio": heavy_ratio,
        "Motorcycle_Ratio": 0.08,
        "Pedestrian_Ratio": 0.02,
        "lat_lon_interaction": lat * lon,
        "lat_squared": lat ** 2,
        "lon_squared": lon ** 2,
        "Nearby_Cluster_Count": min(nearby_cluster, 42),
        "Is_Night": is_night,
        "High_Speed": 1 if speed >= 60 else 0,
        "Speed_Urban": speed * urban_rural,
        "Speed_Night": speed * is_night
    }

    df = pd.DataFrame([data])
    df = df.reindex(columns=model_columns, fill_value=0)
    return df

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
        risk = label_encoder.inverse_transform(prediction)[0]
        return jsonify({"risk_level": risk})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
