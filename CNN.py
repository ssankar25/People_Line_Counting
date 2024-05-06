import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

# Set whether to use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1),  # Input channels = 1, Output chanels = 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Second convolutional layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Third convolutional layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Fourth convolutional layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Fifth convolutional layer
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Determine input size for the fully connected layer based on the feature map size after convolutional layers
        self.fc1 = nn.Linear(64 * 1 * 1, 128)
        self.dropout = nn.Dropout(p=0.5)
        # Final classification layer, outputting two categories
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)  # Flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
