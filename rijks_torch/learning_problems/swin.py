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

        # Global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.side_network = None
        if self.method == "st":
            self.side_network = nn.Sequential(
                nn.Linear(self.model.head.in_features, 512),
                nn.ReLU(),
                nn.Linear(512, n_target_classes)
            )

        freezeLayers(self.model, self.method)
    

    def forward(self, x):
        x = self.model.forward_features(x)  # Extract features
        print(f'Features shape: {x.shape}')  # Debug print

        x = x.permute(0, 3, 1, 2)  # Shape: [batch_size, 1024, 7, 7]
        print(f'After permute: {x.shape}')  # Debug print

        x = self.global_avg_pool(x)  # Apply global average pooling
        print(f'After global avg pool: {x.shape}')  # Debug print

        # Permute back to the original order
        x = x.permute(0, 2, 3, 1)  # Shape: [batch_size, 1, 1, 1024]
        print(f'After permute back: {x.shape}')  # Debug print

        x = torch.flatten(x, 1)  # Flatten the tensor
        print(f'After flattening: {x.shape}')  # Debug print

        main_output = self.model.head(x)  # Classification head
        print(f'Main output shape: {main_output.shape}')  # Debug print
        
        if self.method == "st":
            side_output = self.side_network(x)
            return main_output + side_output
        
        return main_output
    
    def predict_proba(self, x):
        return torch.nn.functional.softmax(self.forward(x), dim=1)

