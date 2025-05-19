# storm/views.py or storm/services.py
import torch
from .pinn_model.model import PINN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PINN().to(device)
model.load_state_dict(torch.load("storm/pinn_model/pinn_model.pt", map_location=device))
model.eval()
