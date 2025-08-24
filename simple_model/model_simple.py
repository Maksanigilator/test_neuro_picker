import torch
import torch.nn as nn
import torch.nn.functional as F

class PickerNet(nn.Module):
    def __init__(self, crop_size=64):
        super().__init__()
        # Input: 2 channels (depth + mask), crop_size x crop_size
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # After 2 pools: crop_size // 4
        fc_input_size = (crop_size // 4) ** 2 * 128
        self.fc1 = nn.Linear(fc_input_size, 256)
        self.fc2 = nn.Linear(256, 1)  # Output: logit for sigmoid

    def forward(self, inputs):
        # inputs: batch_size x 2 x crop_size x crop_size (depth + mask)
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits  # Use sigmoid for probability in loss/inference