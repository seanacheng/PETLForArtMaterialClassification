import torch
from torch import nn
import pandas as pd
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
    
    materials = pd.read_csv('data_annotations/all-hist.csv')["material"].to_list()

    total_per_class = {}
    correct_per_class = {}

    with torch.no_grad():
        for x, y in test_loader:

            logits = model(x.to(device))
            pred_y = torch.argmax(logits, dim=1)
            
            all_preds.extend(pred_y.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            for index, pred in zip(y.cpu().numpy(), pred_y.cpu().numpy()):
                label = materials[index]
                if label not in total_per_class:
                    total_per_class[label] = 0
                    correct_per_class[label] = 0

                total_per_class[label] += 1
                if index == pred:
                    correct_per_class[label] += 1

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    per_class_accuracy = {}
    for label in total_per_class:
        class_acc = 0
        if total_per_class[label] != 0:
            class_acc = correct_per_class[label] / total_per_class[label]
        per_class_accuracy[label] = class_acc
    per_class_accuracy['balanced accuracy'] = balanced_acc

    return balanced_acc, per_class_accuracy
