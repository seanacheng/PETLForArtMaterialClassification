import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score

def test(model: nn.Module, test_loader):
    """
    Function for testing a model
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    correct = 0
    with torch.no_grad():
        for x, y in test_loader:

            logits = model(x.to(device))
            pred_y = torch.argmax(logits, dim=1)

            correct += torch.sum(pred_y == y.to(device)).item()
            
            all_preds.extend(pred_y.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = correct / len(test_loader.dataset)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    return accuracy, balanced_acc
