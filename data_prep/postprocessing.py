import pandas as pd
import matplotlib.pyplot as plt

def create_visualization(arch, method):
    # Load the CSV file
    df = pd.read_csv('results/{}_{}.csv'.format(arch, method))
    # Group by epoch and compute mean and standard deviation
    grouped = df.groupby('epoch').agg(['mean', 'std'])
    # Extract epochs
    epochs = grouped.index

    plot_with_std(epochs, grouped['total_loss']['mean'], grouped['total_loss']['std'], '--', 'b', 'Total Loss')
    plot_with_std(epochs, grouped['train_err']['mean'], grouped['train_err']['std'], '-', 'b', 'Train Error')
    plot_with_std(epochs, grouped['val_loss']['mean'], grouped['val_loss']['std'], '--', 'r', 'Validation Loss')
    plot_with_std(epochs, grouped['val_err']['mean'], grouped['val_err']['std'], '-', 'r', 'Validation Error')

    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.title('{} {} Mean and Standard Deviation'.format(arch, method))
    plt.savefig('results/{}_{}.jpg'.format(arch, method))
    plt.show()

# Plotting function
def plot_with_std(x, y_mean, y_std, linestyle, color, label):
    plt.plot(x, y_mean, linestyle=linestyle, color=color, label=label)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)

create_visualization("ViT", "LP")
create_visualization("ViT", "FT")
create_visualization("ViT", "ST")
create_visualization("Swin", "LP")
create_visualization("Swin", "FT")
create_visualization("Swin", "ST")