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
    train_loader, val_loader, test_loader = RijksDataloader.make_data_loaders(batch_size=64, transform=defs.buildTransform(imnet_norm=True))

    seeds = [17, 204, 596]
    lrs = [0.001, 0.01, 0.1]
    l2pen = 0.1
    epochs = 100
    df = pd.DataFrame(columns=['total_loss', 'train_loss', 'train_err', 'val_loss', 'val_err', 'epoch'])

    figures = []
    
    for model in ["vit", "swin"]:
        for lr in lrs:
            for seed in seeds:
                # Generate a new pretrained model every time
                pretrained_model = None
                if model == "vit":
                    pretrained_model = ViTModel(method="ft")
                elif model == "swin":
                    pretrained_model = SwinModel(method="ft")
                trained_model, results = train(pretrained_model, train_loader, val_loader, lr, epochs, seed, l2pen)

                # Testing model that performed best on validation set:
                final_acc = test(trained_model, test_loader) 
                print(f"final accuracy: {final_acc}")
                
                temp_df = pd.DataFrame({
                    'total_loss': results['tr']['loss'],
                    'train_loss': results['tr']['xent'],
                    'train_err':  results['tr']['err'],
                    'val_loss':   results['va']['xent'],
                    'val_err':    results['va']['err'],
                    'epoch':      results['epochs'],
                    'lr':         lr,
                    'seed':       seed
                })
                fig, ax = plt.subplots(figsize=(9,4))
                ax.plot(results['epochs'], results['tr']['loss'], '--', color='b', label='tr loss')
                ax.plot(results['epochs'], results['tr']['err'], '-', color='b', label='tr err')

                ax.plot(results['epochs'], results['va']['xent'], '--', color='r', label='va xent')
                ax.plot(results['epochs'], results['va']['err'], '-', color='r', label='va err')
                ax.set_title(f'{model} FT test accuracy={final_acc}\nlr={lr}, seed={seed}')
                ax.legend()
                figures.append(fig)
                fig.savefig(f'results/{model}_ft_lr_{lr}_seed_{seed}.png')

                df = pd.concat([df, temp_df], ignore_index=True)

        df.to_csv(f'results/{model}_ft_lr_{lr}_seed_{seed}.csv')

    for fig in figures:
        fig.show()


if __name__ == "__main__":
    main()