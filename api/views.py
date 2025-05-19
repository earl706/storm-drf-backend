from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Alert
from .serializers import AlertSerializer
import torch
from .pinn_model.model import PINN


# storm/views.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = 27  # Ensure this matches your training features
model = PINN(input_dim=input_dim)
model.load_state_dict(
    torch.load("api/pinn_model/pinn_model_rain.pt", map_location="cpu")
)
model.eval()


class AlertAPIView(APIView):
    def get(self, request):
        alerts = Alert.objects.all().order_by("-timestamp")[:50]
        serializer = AlertSerializer(alerts, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = AlertSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PINNPredictView(APIView):
    def post(self, request):
        # Extract input from request
        input_data = request.data.get("inputs")  # expects a list or list of lists

        # Convert to tensor
        x = torch.tensor(input_data, dtype=torch.float32)

        # Load model
        model = PINN()
        model.load_state_dict(
            torch.load("api/pinn_model/pinn_model_rain.pt", map_location="cpu")
        )
        model.eval()

        # Predict
        with torch.no_grad():
            prediction = model(x).numpy().tolist()

        return Response({"prediction": prediction}, status=status.HTTP_200_OK)
