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

# Weather API Key from Render Environment
API_KEY = os.environ.get("API_KEY")

if not API_KEY:
    raise ValueError("API_KEY not found. Please set it in your Render Environment Variables.")

# ----------------------------
# 2. Real-Time Data Helpers
# ----------------------------

def get_realtime_data(lat, lon):
    """Fetches Weather (OWM) and Road Metadata (Overpass)."""
    
    # --- Part A: Weather & Light (OpenWeatherMap) ---
    weather_code, is_night, road_surface = 1, 0, 1  # Defaults
    weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}"
    
    try:
        w_res = requests.get(weather_url, timeout=5).json()
        main_weather = w_res["weather"][0]["main"]
        
        # Determine if it is night based on sunrise/sunset timestamps
        now = datetime.now().timestamp()
        is_night = 1 if (now < w_res['sys']['sunrise'] or now > w_res['sys']['sunset']) else 0
        
        # Infer Road Surface (Wet if raining/snowing)
        if main_weather in ["Rain", "Drizzle", "Thunderstorm", "Snow"]:
            road_surface = 2 
        
        # Map to your model's codes
        weather_map = {"Clear": 1, "Clouds": 2, "Rain": 3, "Drizzle": 3, "Mist": 4, "Fog": 4}
        weather_code = weather_map.get(main_weather, 1)
    except:
        print("Weather API failed, using defaults.")

    # --- Part B: Road Metadata (Overpass API) ---
    speed_limit, urban_rural, road_type = 40, 1, 1  # Defaults
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    way(around:50, {lat}, {lon})["highway"];
    out tags;
    """
    try:
        o_res = requests.get(overpass_url, params={'data': overpass_query}, timeout=10).json()
        if o_res['elements']:
            tags = o_res['elements'][0]['tags']
            # Get maxspeed or assign based on road category
            speed_val = tags.get("maxspeed", "40")
            speed_limit = int(''.join(filter(str.isdigit, speed_val))) if any(c.isdigit() for c in speed_val) else 40
            
            # Map Highway types to Urban (1) or Rural (2)
            highway = tags.get("highway", "residential")
            urban_rural = 2 if highway in ["motorway", "trunk", "primary"] else 1
    except:
        print("Overpass API failed, using defaults.")

    return weather_code, is_night, road_surface, speed_limit, urban_rural, road_type


# ----------------------------
# 3. Feature Engineering
# ----------------------------

def generate_features(lat, lon):
    # Fetch dynamic data
    weather, is_night, surface, speed, urban_rural, road_type = get_realtime_data(lat, lon)
    
    # Mathematical Transformations
    now = datetime.now()
    hour = now.hour
    lat_lon_interaction = lat * lon
    lat_squared = lat ** 2
    lon_squared = lon ** 2

    # Realistic Averages based on your training data to avoid bias
    # These prevent the "Always High Risk" issue caused by high fixed ratios
    heavy_ratio = 0.05
    motorcycle_ratio = 0.08
    pedestrian_ratio = 0.02
    nearby_cluster = 1

    # Interaction Features
    high_speed = 1 if speed >= 60 else 0
    speed_urban = speed * urban_rural
    speed_night = speed * is_night

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
    
    # Ensure all One-Hot columns are present and ordered
    df = df.reindex(columns=model_columns, fill_value=0)
    
    return df

# ----------------------------
# 4. Routes
# ----------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    lat = float(data["latitude"])
    lon = float(data["longitude"])

    # Generate the 20 features required by the XGBoost model
    features = generate_features(lat, lon)

    # Make Prediction
    prediction = model.predict(features)
    risk = label_encoder.inverse_transform(prediction)[0]

    return jsonify({
        "risk_level": risk
    })

if __name__ == "__main__":
    app.run(debug=True)
