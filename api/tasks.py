from celery import shared_task
import requests
import torch
from .pinn_model.model import PINN
from django.utils import timezone
from .models import Alert
import numpy as np

norm_stats = np.load("api/pinn_model/normalization_stats.npz")
mean = norm_stats["mean"]
std = norm_stats["std"]


# Alert classification thresholds
def classify_alerts(prob_rain, prob_flood, water_level):
    alerts = {}

    if prob_rain >= 0.20:
        if prob_rain >= 0.90:
            alerts["RAINFALL"] = ("SEVERE", prob_rain)
        elif prob_rain >= 0.70:
            alerts["RAINFALL"] = ("HIGH", prob_rain)
        elif prob_rain >= 0.40:
            alerts["RAINFALL"] = ("MODERATE", prob_rain)

    if prob_flood >= 0.10:
        if prob_flood >= 0.85:
            alerts["FLOOD"] = ("SEVERE", prob_flood)
        elif prob_flood >= 0.60:
            alerts["FLOOD"] = ("HIGH", prob_flood)
        elif prob_flood >= 0.30:
            alerts["FLOOD"] = ("MODERATE", prob_flood)

    if water_level >= 0.3:
        if water_level >= 3.0:
            alerts["WATER_LEVEL"] = ("SEVERE", water_level)
        elif water_level >= 1.5:
            alerts["WATER_LEVEL"] = ("HIGH", water_level)
        elif water_level >= 0.7:
            alerts["WATER_LEVEL"] = ("MODERATE", water_level)

    return alerts


@shared_task
def fetch_weather_and_predict():
    lat, lon = 8.4822, 124.6472
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&current=temperature_2m,soil_temperature_0_to_7cm,soil_temperature_7_to_28cm,"
        "soil_temperature_28_to_100cm,soil_temperature_100_to_255cm,"
        "soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,soil_moisture_28_to_100cm,soil_moisture_100_to_255cm,"
        "pressure_msl,surface_pressure,cloud_cover,cloud_cover_low,cloud_cover_mid,cloud_cover_high,"
        "et0_fao_evapotranspiration,vapour_pressure_deficit,weather_code,wind_gusts_10m,"
        "wind_direction_100m,wind_direction_10m,wind_speed_100m,wind_speed_10m,"
        "precipitation,apparent_temperature,dew_point_2m,relative_humidity_2m"
        "&timezone=auto"
    )

    response = requests.get(url)
    data = response.json()
    current = data.get("current", {})

    expected_keys = [
        "temperature_2m",
        "soil_temperature_0_to_7cm",
        "soil_temperature_7_to_28cm",
        "soil_temperature_28_to_100cm",
        "soil_temperature_100_to_255cm",
        "soil_moisture_0_to_7cm",
        "soil_moisture_7_to_28cm",
        "soil_moisture_28_to_100cm",
        "soil_moisture_100_to_255cm",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "et0_fao_evapotranspiration",
        "vapour_pressure_deficit",
        "weather_code",
        "wind_gusts_10m",
        "wind_direction_100m",
        "wind_direction_10m",
        "wind_speed_100m",
        "wind_speed_10m",
        "precipitation",
        "apparent_temperature",
        "dew_point_2m",
        "relative_humidity_2m",
    ]

    features = [float(current.get(key, 0.0) or 0.0) for key in expected_keys]
    features_norm = (np.array(features) - mean) / std
    input_tensor = torch.tensor([features_norm], dtype=torch.float32)

    device = torch.device("cpu")
    input_dim = len(expected_keys)
    model = PINN(input_dim=input_dim).to(device)
    model.load_state_dict(
        torch.load("api/pinn_model/pinn_model_rain.pt", map_location=device)
    )
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)[0]
        prob_rain = torch.sigmoid(output[0]).item()
        prob_flood = torch.sigmoid(output[1]).item()
        water_level = output[2].item()

    print(f"🌧️ Rainfall Probability: {prob_rain:.4f}")
    print(f"🌊 Flood Probability: {prob_flood:.4f}")
    print(f"📏 Water Level: {water_level:.4f} meters")

    alerts = classify_alerts(prob_rain, prob_flood, water_level)

    for alert_type, (severity_label, severity_value) in alerts.items():
        if severity_label in ("MODERATE", "HIGH", "SEVERE"):
            Alert.objects.create(
                alert_type=alert_type,
                source="PINN",
                severity=severity_value,
                message=f"{alert_type.title()} level: {severity_label} ({severity_value:.2f})",
                metadata={"level": severity_label, "value": severity_value},
                latitude=lat,
                longitude=lon,
                timestamp=timezone.now(),
            )
