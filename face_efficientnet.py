import torch
import torch.nn as nn
from torchvision import models


class FaceEfficientNet(nn.Module):
    """
    EfficientNet-B3 forensic detector
    Must match training architecture EXACTLY
    """

    def __init__(self):
        super().__init__()

        # IMPORTANT: do NOT wrap inside self.model
        self.features_model = models.efficientnet_b3(weights="IMAGENET1K_V1")

        in_features = self.features_model.classifier[1].in_features

        self.features_model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.features_model(x).squeeze(1)

    def forward_logits(self, x):
        return self.forward(x)
