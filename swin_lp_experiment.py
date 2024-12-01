import torch
from rijks_torch.learning_problems import SwinModel
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

    lrs = [0.001, 0.01]
    seeds = [17, 596, 2043]
    l2pen = 0.01  # simplify training
    epochs = 100

    first_run = True
    best_acc = 0
    for lr in lrs:
        for seed in seeds:
            pretrained_model = SwinModel(method="lp", seed=seed)
            print("lr: {}, seed: {}".format(lr, seed))
            # Training and validating (best model on val set returned):
            trained_model, results = train(pretrained_model, train_loader, val_loader, lr, epochs, seed, l2pen)

            # Testing model that performed best on validation set:
            accuracy, balanced_acc = test(trained_model, test_loader) 
            print("final accuracy: {}, balanced accuracy: {}".format(accuracy, balanced_acc))
            if balanced_acc > best_acc:
                best_acc = balanced_acc
                # torch.save(trained_model.state_dict(), "results/best_Swin_LP_model.pth")

            df = pd.DataFrame({
                'total_loss': results['tr']['loss'],
                'train_loss': results['tr']['xent'],
                'train_err':  results['tr']['err'],
                'val_loss':   results['va']['xent'],
                'val_err':    results['va']['err'],
                'epoch':      results['epochs'],
                'seed':       results['seed'],
                'lr':         results['lr'],
            })
            if first_run:
                df.to_csv('results/Swin_LP.csv', mode="w", index=False, header=True) # overwrites if file already exists
                first_run = False
            else:
                df.to_csv('results/Swin_LP.csv', mode="a", index=False, header=False)

            # plt.figure()
            # plt.plot(results['epochs'], results['tr']['loss'], '--', color='b', label='tr loss')
            # plt.plot(results['epochs'], results['tr']['err'], '-', color='b', label='tr err')

            # plt.plot(results['epochs'], results['va']['xent'], '--', color='r', label='va xent')
            # plt.plot(results['epochs'], results['va']['err'], '-', color='r', label='va err')
            # plt.title('Swin LP\nlr={}, seed={}'.format(lr, seed))
            # plt.legend()
            # plt.savefig('results/Swin_LP_lr{}_seed{}.jpg'.format(str(lr)[2:], seed))
            # plt.close()


if __name__ == "__main__":
    main()