
import torch
import torch.nn as nn

class PickerNet(nn.Module):
    def __init__(self, in_features=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        B,N,F = x.shape
        y = self.net(x.view(B*N, F))
        return y.view(B, N)
