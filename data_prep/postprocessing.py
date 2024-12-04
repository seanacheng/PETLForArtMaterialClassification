import pandas as pd
import matplotlib.pyplot as plt

def create_bal_acc_visualization(arch):
    lp_df = pd.read_csv('results/{}_LP_hyperparameter_search.csv'.format(arch))
    ft_df = pd.read_csv('results/{}_FT_hyperparameter_search.csv'.format(arch))
    st_df = pd.read_csv('results/{}_ST_hyperparameter_search.csv'.format(arch))
    # Group by lr and compute mean and standard deviation
    lp_grouped = lp_df.groupby('lr').agg(['mean', 'std'])
    ft_grouped = ft_df.groupby('lr').agg(['mean', 'std'])
    st_grouped = st_df.groupby('lr').agg(['mean', 'std'])

    plt.figure()
    plot_with_std(lp_grouped.index, lp_grouped['balanced_acc']['mean'], lp_grouped['balanced_acc']['std'], '-', 'r', 'Linear Probing', "lightcoral")
    plot_with_std(ft_grouped.index, ft_grouped['balanced_acc']['mean'], ft_grouped['balanced_acc']['std'], '-', 'b', 'Full Fine-Tuning', "lightblue")
    plot_with_std(st_grouped.index, st_grouped['balanced_acc']['mean'], st_grouped['balanced_acc']['std'], '-', 'g', 'Side-Tuning', "lightgreen")

    # Add labels and legend
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Balanced Accuracy')
    plt.legend()
    plt.title('{} LR vs. Bal. Acc.'.format(arch))
    plt.savefig('results/{}_lr_acc.jpg'.format(arch))
    plt.show()


# Plotting function
def plot_with_std(x, y_mean, y_std, linestyle, color, label, fill_color):
    plt.plot(x, y_mean, linestyle=linestyle, color=color, label=label)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=fill_color, alpha=0.2)

create_bal_acc_visualization("ViT")
create_bal_acc_visualization("Swin")