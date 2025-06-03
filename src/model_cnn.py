# src/model_cnn.py

import torch
import torch.nn as nn
from torchvision import models

# 1. Custom CNN model
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112x112
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56x56
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28x28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 2. Transfer learning using EfficientNet-B0
def get_efficientnet_model(pretrained=True):
    model = models.efficientnet_b0(pretrained=pretrained)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)  # Binary classification
    return model

# 3. Helper function to select model
def get_model(model_name='custom', pretrained=True):
    if model_name == 'custom':
        return CustomCNN()
    elif model_name == 'efficientnet':
        return get_efficientnet_model(pretrained)
    else:
        raise ValueError("Model not supported. Choose 'custom' or 'efficientnet'.")
