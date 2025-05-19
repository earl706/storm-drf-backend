import torch

checkpoint = torch.load("api/pinn_model/pinn_model_rain.pt", map_location="cpu")
for key, tensor in checkpoint.items():
    print(f"{key}: {tuple(tensor.shape)}")
