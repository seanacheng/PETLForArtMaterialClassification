import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_loss_visualization_subplot(ax, arch, method):
    df = pd.read_csv(f'results/{arch}_{method}.csv')
    grouped = df.groupby('epoch').agg(['mean', 'std'])
    epochs = grouped.index

    plot_with_std(ax, epochs, grouped['total_loss']['mean'], grouped['total_loss']['std'], '--', 'b', 'Total Loss')
    plot_with_std(ax, epochs, grouped['train_err']['mean'], grouped['train_err']['std'], '-', 'b', 'Train Error')
    plot_with_std(ax, epochs, grouped['val_loss']['mean'], grouped['val_loss']['std'], '--', 'r', 'Validation Loss')
    plot_with_std(ax, epochs, grouped['val_err']['mean'], grouped['val_err']['std'], '-', 'r', 'Validation Error')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend()
    ax.set_title(f'{arch} {method} Loss Per Epoch')

def plot_with_std(ax, x, y_mean, y_std, linestyle, color, label):
    ax.plot(x, y_mean, linestyle=linestyle, color=color, label=label)
    ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=0.2)

def create_loss_plots(arch):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    create_loss_visualization_subplot(axes[0], arch, "LP")
    create_loss_visualization_subplot(axes[1], arch, "FT")
    create_loss_visualization_subplot(axes[2], arch, "ST")

    plt.tight_layout()
    plt.savefig('results/{}_loss_per_epoch.jpg'.format(arch))

create_loss_plots("ViT")
create_loss_plots("Swin")

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


def plot_with_std(x, y_mean, y_std, linestyle, color, label, fill_color):
    plt.plot(x, y_mean, linestyle=linestyle, color=color, label=label)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, color=fill_color, alpha=0.2)

create_bal_acc_visualization("ViT")
create_bal_acc_visualization("Swin")

def plot_class_accuracies(arch):
    lp_df = pd.read_csv('results/{}_LP_acc_per_class.csv'.format(arch))
    ft_df = pd.read_csv('results/{}_FT_acc_per_class.csv'.format(arch))
    st_df = pd.read_csv('results/{}_ST_acc_per_class.csv'.format(arch))

    # Calculate the mean for each column
    ft_averages = ft_df.mean()
    lp_averages = lp_df.mean()
    st_averages = st_df.mean()
    ft_std = ft_df.std()
    lp_std = lp_df.std()
    st_std = st_df.std()

    categories = ft_averages.index
    n_categories = len(categories)

    # X-axis positions for groups of bars
    x = np.arange(n_categories) * 2

    # Bar width
    bar_width = 0.5

    # Plot the averages as a bar chart
    plt.figure(figsize=(16,6))
    plt.bar(x - bar_width, lp_averages.values, bar_width, yerr=lp_std.values, capsize=5, color='r', label='Linear Probing')
    plt.bar(x, ft_averages.values, bar_width, yerr=ft_std.values, capsize=5, color='b', label='Full Fine-Tuning')
    plt.bar(x + bar_width, st_averages.values, bar_width, yerr=st_std.values, capsize=5, color='g', label='Side-Tuning')

    # Add labels and title
    plt.ylabel('Test Accuracy')
    plt.title('{} Test Accuracy Per Class'.format(arch))
    plt.xticks(x, ft_averages.index, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/{}_acc_per_class.jpg'.format(arch))

plot_class_accuracies("ViT")
plot_class_accuracies("Swin")
