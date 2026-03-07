from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import os

app = Flask(__name__)

# ----------------------------
# 1. Load Model Files
# ----------------------------
model = joblib.load("accident_risk_model.pkl")
model_columns = joblib.load("model_columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")

API_KEY = os.environ.get("API_KEY")

# ----------------------------
# 2. Scenario Logic (The Bridge)
# ----------------------------
def get_scenario_ratios(highway_tag, urban_rural, speed):
    """
    Translates real-world road types into the specific ratios 
    your model saw during training.
    """
    # Baseline: Low Risk averages from your Colab stats
    heavy, motor, ped, clusters = 0.712, 0.05, 0.07, 27

    # Scenario: High-Speed Highways/Rural Links
    if urban_rural == 2 or speed >= 70 or highway_tag in ["motorway", "trunk", "primary"]:
        heavy = 0.73  # Slightly higher than average for danger signal
        ped = 0.02    # Low pedestrian presence
        clusters = 36 # Moving toward 'High Risk' cluster average (37.8)

    # Scenario: Urban Residential/City Streets
    elif highway_tag in ["residential", "living_street", "tertiary"]:
        heavy = 0.45  # Lower heavy vehicle ratio for city streets
        ped = 0.15    # Higher pedestrian presence
        clusters = 26 # Keeping it near the 'Low Risk' average

    return heavy, motor, ped, clusters

# ----------------------------
# 3. Real-Time Data Fetching
# ----------------------------
def get_realtime_data(lat, lon):
    # --- Weather & Light ---
    weather_code, is_night, surface = 1, 0, 1
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}"
    try:
        w_res = requests.get(weather_url, timeout=5).json()
        main_weather = w_res["weather"][0]["main"]
        now = datetime.now().timestamp()
        is_night = 1 if (now < w_res['sys']['sunrise'] or now > w_res['sys']['sunset']) else 0
        if main_weather in ["Rain", "Drizzle", "Thunderstorm", "Snow"]: surface = 2
        
        weather_map = {"Clear": 1, "Clouds": 2, "Rain": 3, "Mist": 4}
        weather_code = weather_map.get(main_weather, 1)
    except: pass

    # --- Road Metadata (Overpass) ---
    speed, urban_rural, highway_tag = 40, 1, "residential"
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f'[out:json];way(around:50, {lat}, {lon})["highway"];out tags;'
    try:
        o_res = requests.get(overpass_url, params={'data': overpass_query}, timeout=10).json()
        if o_res['elements']:
            tags = o_res['elements'][0]['tags']
            highway_tag = tags.get("highway", "residential")
            speed_val = tags.get("maxspeed", "40")
            speed = int(''.join(filter(str.isdigit, speed_val))) if any(c.isdigit() for c in speed_val) else 40
            urban_rural = 2 if highway_tag in ["motorway", "trunk", "primary"] else 1
    except: pass

    return weather_code, is_night, surface, speed, urban_rural, highway_tag

# ----------------------------
# 4. Feature Engineering
# ----------------------------
def generate_features(lat, lon):
    # Fetch dynamic data
    weather, is_night, surface, speed, urban_rural, highway_tag = get_realtime_data(lat, lon)
    
    # Get Scenario Ratios based on the API results
    heavy, motor, ped, clusters = get_scenario_ratios(highway_tag, urban_rural, speed)
    
    # Mathematical Transformations
    now = datetime.now()
    hour = now.hour

    data = {
        "Speed_limit": speed,
        "Urban_or_Rural_Area": urban_rural,
        "Road_Type": 1, # Defaulting to Single Carriageway
        "Weather_Conditions": weather,
        "Light_Conditions": is_night,
        "Road_Surface_Conditions": surface,
        "Hour": hour,
        "latitude": lat,
        "longitude": lon,
        "Heavy_Vehicle_Ratio": heavy,
        "Motorcycle_Ratio": motor,
        "Pedestrian_Ratio": ped,
        "lat_lon_interaction": lat * lon,
        "lat_squared": lat ** 2,
        "lon_squared": lon ** 2,
        "Nearby_Cluster_Count": clusters,
        "Is_Night": is_night,
        "High_Speed": 1 if speed >= 60 else 0,
        "Speed_Urban": speed * urban_rural,
        "Speed_Night": speed * is_night
    }

    df = pd.DataFrame([data])
    df = df.reindex(columns=model_columns, fill_value=0)
    return df

# ----------------------------
# 5. Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.json
#     lat, lon = float(data["latitude"]), float(data["longitude"])
#     features = generate_features(lat, lon)
#     prediction = model.predict(features)
#     risk = label_encoder.inverse_transform(prediction)[0]
#     return jsonify({"risk_level": risk})

# if __name__ == "__main__":
#     app.run(debug=True)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    lat, lon = float(data["latitude"]), float(data["longitude"])
    
    # Debugging: Get the data first to see what's happening
    weather, is_night, surface, speed, urban_rural, highway_tag = get_realtime_data(lat, lon)
    heavy, motor, ped, clusters = get_scenario_ratios(highway_tag, urban_rural, speed)
    
    print(f"--- DEBUG DATA for {lat}, {lon} ---")
    print(f"Road Tag: {highway_tag} | Speed: {speed}")
    print(f"Assigned Clusters: {clusters} | Assigned Heavy Ratio: {heavy}")
    
    features = generate_features(lat, lon)
    prediction = model.predict(features)
    risk = label_encoder.inverse_transform(prediction)[0]
    
    print(f"FINAL PREDICTION: {risk}")
    return jsonify({"risk_level": risk})
