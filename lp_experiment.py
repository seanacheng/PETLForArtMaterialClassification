import torch
from rijks_torch.learning_problems import ViTModel, SwinModel
import rijks_torch.learning_problems.defaults as defs
from rijks_torch.data_loading.rijksdataloader import RijksDataloader
from rijks_torch.training import train, test
import matplotlib.pyplot as plt


def main():
    
    # If there is no gpu, we aren't actually going to run it..
    assert torch.cuda.is_available(), "There was no GPU :-("

    # Creating the dataloaders from given arguments:
    train_loader, val_loader, test_loader = RijksDataloader.make_data_loaders(batch_size=128, transform=defs.buildTransform(imnet_norm=True))

    # Get the model tailored to specification. Using getattr because function from cli args
    model = ViTModel(method="lp")

    seed = 1
    lr = 0.01
    epochs = 100
    # Training and validating (best model on val set returned):
    model, results = train(model, train_loader, val_loader, lr, epochs, seed)

    # Testing model that performed best on validation set:
    print(test(model, test_loader))

    plt.plot(results['epochs'], results['tr']['loss'], '--', color='b', label='tr loss')
    plt.plot(results['epochs'], results['tr']['err'], '-', color='b', label='tr err')

    plt.plot(results['epochs'], results['va']['xent'], '--', color='r', label='va xent')
    plt.plot(results['epochs'], results['va']['err'], '-', color='r', label='va err')
    plt.legend()

    model_swin = SwinModel("lp")

if __name__ == "__main__":
    main()