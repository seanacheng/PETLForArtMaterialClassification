import torch
from torch import nn
import pandas as pd
from ..data_loading import RijksDataloaders

def test(model: nn.Module, test_loader):
    """
    ### Tests the given model on dataloaders.test.
    Saves the following data as csv-files:\n
    \t(1) '{name}-test_predictions.csv' gives the full softmax prediction, as well as the correct output;\n
    \t(2) '{name}-test_confusion.csv' gives the confusion matrix.\n
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            pred_y = torch.argmax(logits, dim=1)
            correct += torch.sum(pred_y == y.to(device)).item()
    return correct / len(test_loader.dataset)
