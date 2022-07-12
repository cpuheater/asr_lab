import torch
import torch.nn as nn
import torch.nn.functional as F

class Cnn(nn.Module):
    def __init__(self, channels: int, output_size: int):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(7680, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x) -> torch.Tensor:
        x = self.conv1(x)
        x = F.max_pool2d(F.relu(x), kernel_size=3)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x