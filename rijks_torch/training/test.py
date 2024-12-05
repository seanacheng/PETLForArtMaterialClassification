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
    
    materials = [
        'papier',
        'zilver',
        'faience',
        'porselein',
        'hout',
        'brons',
        'glas (materiaal)',
        'perkament',
        'geprepareerd papier',
        'fotopapier',
        'ijzer',
        'Japans papier',
        'ivoor',
        'Oosters papier',
        'eikenhout'
    ]

    total_per_class = {material: 0 for material in materials}
    correct_per_class = {material: 0 for material in materials}

    with torch.no_grad():
        for x, y in test_loader:

            logits = model(x.to(device))
            pred_y = torch.argmax(logits, dim=1)
            
            all_preds.extend(pred_y.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

            for label, pred in zip(y.cpu().numpy(), pred_y.cpu().numpy()):
                if label in total_per_class:
                    total_per_class[label] += 1
                else:
                    total_per_class[label] == 1
                    correct_per_class[label] == 0

                if label == pred:
                    correct_per_class[label] += 1

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)

    per_class_accuracy = {
        label: correct_per_class[label] / total_per_class[label]
        for label in total_per_class
        }


    return balanced_acc, per_class_accuracy
