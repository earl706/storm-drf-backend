from django.urls import path, include
from .views import AlertAPIView, PINNPredictView

urlpatterns = [
    path("alerts/", AlertAPIView.as_view(), name="alerts-basic"),
    path("predict/", PINNPredictView.as_view(), name="pinn-predict"),
]
