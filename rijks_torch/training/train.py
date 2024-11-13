import torch
from torch import nn, optim
import sklearn.metrics
import numpy as np

import tqdm
from copy import deepcopy

def train(model: nn.Module, train_loader, val_loader, lr=0.01, num_epochs=20, seed=12, l2pen=0.0):
    """
    Function for training a model
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Allocate lists for tracking progress each epoch
    tr_info = {'xent':[], 'err':[], 'loss':[]}
    va_info = {'xent':[], 'err':[]}
    epochs = []

    # Keeping track of the best model:
    best_model = deepcopy(model.state_dict())
    best_va_loss = float('inf')
    best_epoch = 0

    early_stop_after_epochs = 20
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    progressbar = tqdm.tqdm(range(num_epochs + 1))
    pbar_info = {}

    n_train = float(len(train_loader.dataset))
    n_batch_tr = float(len(train_loader))
    n_valid = float(len(val_loader.dataset))

    for epoch in progressbar:
        if epoch > 0:
            model.train()
            tr_loss = 0.0  # aggregate total loss
            tr_xent = 0.0  # aggregate cross-entropy
            tr_err = 0     # count mistakes on train set
            pbar_info['batch_done'] = 0
            for batchnum, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()

                # Forward pass:
                logits = model(x.to(device))
                
                xent_loss_func = nn.CrossEntropyLoss(reduction='mean')
                loss_xent = xent_loss_func(logits, y.to(device))

                loss_l2 = 0
                for param in model.parameters():
                    if param.requires_grad:
                        loss_l2 += torch.sum(param ** 2)

                # Backward pass:
                loss = loss_xent + l2pen / n_train * loss_l2
                loss.backward()
                optimizer.step()

                pbar_info['batch_done'] += 1        
                progressbar.set_postfix(pbar_info)

                # Increment loss metrics we track for debugging/diagnostics
                tr_loss += loss.item() / n_batch_tr
                tr_xent += loss_xent.item() / n_batch_tr
                tr_err += sklearn.metrics.zero_one_loss(
                    logits.argmax(axis=1).detach().cpu().numpy(), y, normalize=False)
            tr_err_rate = tr_err / n_train
        else:
            # First epoch (0) doesn't train, just measures initial perf on val
            tr_loss = np.nan
            tr_xent = np.nan
            tr_err_rate = np.nan

        # Track performance on val set
        with torch.no_grad():
            model.eval() # In EVAL mode
            va_xent = 0.0
            va_err = 0
            for x_val, y_val in val_loader:
                logits = model(x_val.to(device))
                xent_loss_func = nn.CrossEntropyLoss(reduction='sum')
                va_xent += xent_loss_func(logits, y_val.to(device)).item()

                va_err += sklearn.metrics.zero_one_loss(
                    logits.argmax(axis=1).detach().cpu().numpy(), y_val, normalize=False)
            va_err_rate = va_err / n_valid

        # Update diagnostics and progress bar
        epochs.append(epoch)
        tr_info['loss'].append(tr_loss)
        tr_info['xent'].append(tr_xent)
        tr_info['err'].append(tr_err_rate)        
        va_info['xent'].append(va_xent)
        va_info['err'].append(va_err_rate)
        pbar_info.update({
            "tr_xent": tr_xent, "tr_err": tr_err_rate,
            "va_xent": va_xent, "va_err": va_err_rate,
            })
        progressbar.set_postfix(pbar_info)

        # Save best model and early stop if no new best for too long:
        if va_xent < best_va_loss:
            best_va_loss = va_xent
            best_epoch = epoch
            best_tr_err_rate = tr_err_rate
            best_va_err_rate = va_err_rate
            best_model = deepcopy(model.state_dict())
            torch.save(best_model, f"best_model.pth")
        elif epoch - best_epoch >= early_stop_after_epochs:
            print("Stopped early.")
            break
    print(f"Finished after epoch {epoch}, best epoch={best_epoch}")
    print("best va_xent %.3f" % best_va_loss)
    print("best tr_err %.3f" % best_tr_err_rate)
    print("best va_err %.3f" % best_va_err_rate)

    # Return the best model found:
    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    results = {
        'lr': lr,
        'seed': seed,
        'tr':tr_info,
        'va':va_info,
        'best_tr_err': best_tr_err_rate,
        'best_va_err': best_va_err_rate,
        'best_va_loss': best_va_loss,
        'best_epoch': best_epoch,
        'epochs': epochs}
    return model, results
