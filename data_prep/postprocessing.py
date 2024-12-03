import pandas as pd
import matplotlib.pyplot as plt

def create_visualization(arch, method):
    # Load the CSV file
    df = pd.read_csv('results/{}_{}_hyperparameter_search.csv'.format(arch, method))
    # Group by lr and compute mean and standard deviation
    grouped = df.groupby('lr').agg(['mean', 'std'])
    # Extract epochs
    epochs = grouped.index

    plot_with_std(epochs, grouped['balanced_acc']['mean'], grouped['balanced_acc']['std'], '-', 'b', 'l2pen=0.01')
    
    # Add labels and legend
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Balanced Accuracy')
    plt.legend()
    plt.title('{} {} Mean and Standard Deviation'.format(arch, method))
    plt.savefig('results/{}_{}.jpg'.format(arch, method))
    plt.show()

# Plotting function
def plot_with_std(x, y_mean, y_std, linestyle, color, label):
    plt.plot(x, y_mean, linestyle=linestyle, color=color, label=label)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.2)

# create_visualization("ViT", "LP")
# create_visualization("ViT", "FT")
# create_visualization("ViT", "ST")
# create_visualization("Swin", "LP")
# create_visualization("Swin", "FT")
# create_visualization("Swin", "ST")