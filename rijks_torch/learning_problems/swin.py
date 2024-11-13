from .defaults import freezeLayers
import torch
from torch import nn
import timm

class SwinModel(torch.nn.Module):

    def __init__(self, method, seed: int = 42, n_target_classes: int = 15, pretrained=True):
        super().__init__()

        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)

        if method in ["lp", "ft", "st"]:
            freezeLayers(self.model, method, n_target_classes)
        
        torch.manual_seed(int(seed))
        # Replace head with one that fits the task
        self.model.head = nn.Linear(self.model.head.in_features, n_target_classes)
    

    def forward(self, x):
        return self.model(x)
    
    def predict_proba(self, x):
        return torch.nn.functional.softmax(self.forward(x), dim=1)

