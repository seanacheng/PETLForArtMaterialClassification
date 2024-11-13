
import torch
from rijks_torch.learning_problems import ViTModel, SwinModel
import rijks_torch.learning_problems.defaults as defs
from rijks_torch.data_loading.rijksdataloaders import RijksDataloaders
from rijks_torch.training import train, test
import matplotlib.pyplot as plt


def main():
    """
    Arguments given to script must be:\n
    1) name of the experiment: will be prepended to all output files;\n
    2) name of the model such that 'get_{name}_problem()' is defined;\n
    3) dataset files. e.g. if /x/y/z is given, then
        /x/y/z-train.csv,
        /x/y/z-val.csv,
        /x/y/z-test.csv,
        /x/y/z-hist.csv,must exist;\n
    4) name of directory containing all the jpg files (should be stored locally, so under /local)
    """
    
    # If there is no gpu, we aren't actually going to run it..
    assert torch.cuda.is_available(), "There was no GPU :-("

    # Creating the dataloaders from given arguments:
    train_loader, val_loader, test_loader = RijksDataloaders.make_data_loaders(batch_size=32)

    # Get the model tailored to specification. Using getattr because function from cli args
    model = ViTModel(method="ft")

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
    plt.legend();

if __name__ == "__main__":
    main()