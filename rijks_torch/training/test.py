import torch
from torch import nn

def test(model: nn.Module, test_loader):
    """
    Function for testing a model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    correct = 0
    with torch.no_grad():
        for x, y in test_loader:

            logits = model(x.to(device))
            pred_y = torch.argmax(logits, dim=1)

            correct += torch.sum(pred_y == y.to(device)).item()
            
    return correct / len(test_loader.dataset)
