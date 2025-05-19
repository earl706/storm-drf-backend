from django.db import models


class Alert(models.Model):
    ALERT_SOURCE_CHOICES = [
        ("IOT", "IoT Sensor"),
        ("PINN", "Physics-Informed NN"),
        ("ML", "Machine Learning Model"),
        ("NWP", "Numerical Weather Prediction"),
        ("OWA", "OpenWeatherAPI"),
        ("USER", "Crowdsourced Report"),
    ]

    ALERT_TYPE_CHOICES = [
        ("FLOOD", "Flood"),
        ("RAINFALL", "Heavy Rainfall"),
        ("WATER_LEVEL", "Water Level Spike"),
        ("BLOCKAGE", "Drainage Blockage"),
        ("TYPHOON", "Typhoon Warning"),
    ]

    alert_type = models.CharField(max_length=20, choices=ALERT_TYPE_CHOICES)
    source = models.CharField(max_length=10, choices=ALERT_SOURCE_CHOICES)

    severity = models.FloatField(help_text="Scale from 0 (low) to 1 (critical)")
    message = models.TextField(blank=True, null=True)
    metadata = models.JSONField(default=dict, blank=True)

    # Simple location fields instead of GIS
    latitude = models.FloatField()
    longitude = models.FloatField()
    barangay = models.CharField(max_length=100, blank=True, null=True)

    timestamp = models.DateTimeField(auto_now_add=True)

    image = models.ImageField(upload_to="alert_images/", blank=True, null=True)
    reporter_contact = models.CharField(max_length=100, blank=True, null=True)

    def __str__(self):
        return f"{self.alert_type} from {self.source} at {self.timestamp}"
