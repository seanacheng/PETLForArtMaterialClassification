from .defaults import freezeLayers
import torch
from torch import nn
from torchvision import models
from torchvision.models.vision_transformer import *

class ViTModel(torch.nn.Module):

    def __init__(self, method: str = "lp", seed: int = 42, n_target_classes: int = 15):
        super().__init__()

        self.method = method
        self.model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        torch.manual_seed(seed)
        self.model.heads.head = nn.Linear(768, n_target_classes)

        # Global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.side_network = None
        if self.method == "st":
            self.side_network = nn.Sequential(
                nn.Linear(self.model.heads.head.in_features, 512),
                nn.ReLU(),
                nn.Linear(512, n_target_classes)
            )

        freezeLayers(self.model, self.method, self.side_network)
    

    def forward(self,x):
        x = self.model._process_input(x)  # Extract features
        x = x.permute(0, 2, 1)  # Shape: [batch_size, 768, 196]
        x = self.global_avg_pool(x)  # Apply global average pooling
        x = x.permute(0, 2, 1)  # Shape: [batch_size, 1, 768]
        x = torch.flatten(x, 1)  # Flatten the tensor
        main_output = self.model.heads.head(x)

        if self.method == "st":
            side_output = self.side_network(x)
            return main_output + side_output
        
        return main_output
    
    def predict_proba(self, x):
        return torch.nn.functional.softmax(self.forward(x), dim=1)
