from .defaults import freezeLayers
import torch
from torch import nn
from torchvision import models
from torchvision.models.vision_transformer import *

class ViTModel(torch.nn.Module):

    def __init__(self, method, seed: int = 42, n_target_classes: int = 15):
        super().__init__()

        self.model = None
        self.model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        if method in ["lp", "ft", "st"]:
            freezeLayers(self.model, method, n_target_classes)
        
        torch.manual_seed(int(seed))
        # Replace head with one that fits the task
        self.model.heads.head = nn.Linear(768, n_target_classes)
    

    def forward(self,x):
        return self.model(x)
    
    def predict_proba(self, x):
        return torch.nn.functional.softmax(self.forward(x), dim=1)
