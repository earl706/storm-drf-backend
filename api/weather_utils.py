# app/weather_utils.py
import requests
from .models import Alert
from django.utils.timezone import now

OPENWEATHER_API_KEY = "e9f3750c2256272b1f641e2ace34f0dc"


def fetch_openweather_data(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    response = requests.get(url, params=params)
    return response.json()


def analyze_weather_data(data):
    # Extract key metrics
    rain = data.get("rain", {}).get("1h", 0)
    wind = data["wind"]["speed"]
    temp = data["main"]["temp"]

    # Mock thresholds / call to ML/PINN models
    flood_risk_score = dummy_PINN_predict(rain, wind, temp)

    if flood_risk_score > 0.8:
        return {
            "alert_type": "FLOOD",
            "severity": flood_risk_score,
            "message": f"High flood risk detected with {rain}mm rainfall",
            "lat": data["coord"]["lat"],
            "lon": data["coord"]["lon"],
        }
    return None


def analyze_and_store_weather_data():
    # Set your region’s coordinates (or loop through a list of barangays)
    lat, lon = 14.5995, 120.9842  # Manila as example
    weather_data = fetch_openweather_data(lat, lon)

    analysis = analyze_weather_data(weather_data)
    if analysis:
        Alert.objects.create(
            alert_type=analysis["alert_type"],
            source="NWP",  # or "PINN"
            severity=analysis["severity"],
            message=analysis["message"],
            latitude=analysis["lat"],
            longitude=analysis["lon"],
            timestamp=now(),
        )
