import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, input_dim=27, hidden_dim=128, output_dim=3):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        output = self.net(x)
        rainfall_prob = torch.sigmoid(output[:, 0])  # probability between 0 and 1
        flood_prob = torch.sigmoid(output[:, 1])  # probability between 0 and 1
        water_level = torch.relu(output[:, 2])  # force water level ≥ 0
        return torch.stack([rainfall_prob, flood_prob, water_level], dim=1)
