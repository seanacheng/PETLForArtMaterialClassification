from .defaults import freezeLayers
import torch
from torch import nn
import timm

class SwinModel(torch.nn.Module):

    def __init__(self, method: str = "lp", seed: int = 42, n_target_classes: int = 15):
        super().__init__()

        self.method = method
        self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        torch.manual_seed(int(seed))
        self.model.head = nn.Linear(1024, n_target_classes)

        self.side_network = None
        if self.method == "st":
            self.side_network = nn.Sequential(
                nn.Linear(self.model.head.in_features, 512),
                nn.ReLU(),
                nn.Linear(512, n_target_classes)
            )

        freezeLayers(self.model, self.method)
    

    def forward(self, x):
        main_output = self.model(x)
        
        if self.method == "st":
            side_output = self.side_network(x)
            return main_output + side_output
        
        return self.model(x)
    
    def predict_proba(self, x):
        return torch.nn.functional.softmax(self.forward(x), dim=1)

