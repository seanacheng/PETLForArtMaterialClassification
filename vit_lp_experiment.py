import torch
from rijks_torch.learning_problems import ViTModel
import rijks_torch.learning_problems.defaults as defs
from rijks_torch.data_loading.rijksdataloader import RijksDataloader
from rijks_torch.training import train, test
import matplotlib.pyplot as plt
import pandas as pd


def main():
    
    # If there is no gpu, we aren't actually going to run it..
    assert torch.cuda.is_available(), "There was no GPU :-("

    # Creating the dataloaders from given arguments:
    train_loader, val_loader, test_loader = RijksDataloader.make_data_loaders(batch_size=128, transform=defs.buildTransform(imnet_norm=True))

    seeds = [17, 204]
    lrs = [0.01]
    l2pen = 0.1
    epochs = 100

    for lr in lrs:
        for seed in seeds:
            pretrained_model = ViTModel(method="lp")
            # Training and validating (best model on val set returned):
            trained_model, results = train(pretrained_model, train_loader, val_loader, lr, epochs, seed, l2pen)
            torch.save(trained_model.state_dict(), "results/best_ViT_LP_model.pth")

            # Testing model that performed best on validation set:
            accuracy, balanced_acc = test(trained_model, test_loader) 
            print(f"final accuracy: {accuracy}, balanced accuracy: {balanced_acc}")

            df = pd.DataFrame({
                'total_loss': results['tr']['loss'],
                'train_loss': results['tr']['xent'],
                'train_err':  results['tr']['err'],
                'val_loss':   results['va']['xent'],
                'val_err':    results['va']['err'],
                'epoch':      results['epochs']
            })

            plt.plot(results['epochs'], results['tr']['loss'], '--', color='b', label='tr loss')
            plt.plot(results['epochs'], results['tr']['err'], '-', color='b', label='tr err')

            plt.plot(results['epochs'], results['va']['xent'], '--', color='r', label='va xent')
            plt.plot(results['epochs'], results['va']['err'], '-', color='r', label='va err')
            plt.title(f'ViT LP\nlr={lr}, seed={seed}')
            plt.legend()

            df.to_csv(f'results/ViT_LP_lr_{str(lr)[2:]}_seed_{seed}.csv')

if __name__ == "__main__":
    main()