import torch
from rijks_torch.learning_problems import ViTModel, SwinModel
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

    # Get the model tailored to specification. Using getattr because function from cli args
    model = ViTModel(method="ft")

    seeds = [17, 204, 596]
    lrs = [0.001, 0.01, 0.1]
    l2pen = 0.1
    epochs = 100
    df = pd.DataFrame(columns=['total_loss', 'train_loss', 'train_err', 'val_loss', 'val_err', 'epoch'])

    figures = []
    
    for lr in lrs:
        for seed in seeds:
            model, results = train(model, train_loader, val_loader, lr, epochs, seed, l2pen)
            # Testing model that performed best on validation set:
            final_acc = test(model, test_loader) 
            print(f"final accuracy: {final_acc}")
            
            temp_df = pd.DataFrame({
                'total_loss': results['tr']['loss'],
                'train_loss': results['tr']['xent'],
                'train_err':  results['tr']['err'],
                'val_loss':   results['va']['xent'],
                'val_err':    results['va']['err'],
                'epoch':      results['epochs']
            })
            fig, ax = plt.subplots(figsize=(9,4))
            ax.plot(results['epochs'], results['tr']['loss'], '--', color='b', label='tr loss')
            ax.plot(results['epochs'], results['tr']['err'], '-', color='b', label='tr err')

            ax.plot(results['epochs'], results['va']['xent'], '--', color='r', label='va xent')
            ax.plot(results['epochs'], results['va']['err'], '-', color='r', label='va err')
            ax.set_title(f'ViT FT test accuracy={final_acc}\nlr={lr}, seed={seed}')
            ax.legend()
            figures.append(fig)
            fig.savefig(f'results/vit_ft_lr_{lr}_seed_{seed}.png')

            df = pd.concat([df, temp_df], ignore_index=True)

    df.to_csv(f'results/ViT_ft_results.csv')
    for fig in figures:
        fig.show()


if __name__ == "__main__":
    main()