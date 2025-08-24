import torch
import torch.nn as nn
import torch.nn.functional as F

class PickerNet(nn.Module):
    def __init__(self, crop_size=96, pose_dim=1):
        super().__init__()
        # GQCNN-like: 4 conv (pairs: conv1-2, conv3-4) + pools after pairs
        self.conv1 = nn.Conv2d(2, 32, kernel_size=5, stride=1, padding=2)  # Input: depth + mask
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Flatten: after 2 pools, crop_size//4 x //4 x 256 + pose_dim
        fc_input_size = (crop_size // 4) ** 2 * 256 + pose_dim
        # 3 FC layers
        self.fc1 = nn.Linear(fc_input_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 1)  # Logit

    def forward(self, inputs, poses):
        # inputs: batch x 2 x 96 x 96
        # poses: batch x 1 (normalized z)
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.cat([x, poses], dim=1)  # Fuse pose
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits