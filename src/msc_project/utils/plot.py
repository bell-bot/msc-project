import os
from typing import Tuple
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

def plot_histograms(backdoored_weights, target_weights, backdoored_biases, target_biases, save_path, filename = "parameter_comparison"):

    os.makedirs(save_path, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    
    #Plot weight distribution
    ax1.hist(target_weights, bins=100, alpha=0.5, density=True, range=(-backdoored_weights.max().item(),backdoored_weights.max().item()), label="Target Model")
    ax1.hist(backdoored_weights, bins=100, alpha=0.7, density=True, label="Backdoor")
    ax1.set_title("Weights")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    #ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True)

    #Plot bias distribution
    ax2.hist(target_biases, bins=100, alpha=0.5, density=True, label="Target Model")
    ax2.hist(backdoored_biases, bins=100, alpha=0.7, density=True, label="Backdoor")
    
    ax2.set_title("Biases")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Density")
    #ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True)

    plt.savefig(os.path.join(save_path, filename + ".pdf"))
    plt.savefig(os.path.join(save_path, filename + ".png"))
    plt.close()

def plot_separate_histograms(backdoored_weights, target_weights, backdoored_biases, target_biases, save_path):

    os.makedirs(save_path, exist_ok=True)

    def plot_and_save(data, title, filename):
        plt.figure(figsize=(10, 6))
        plt.hist(data.numpy(), bins=100, alpha=0.7, color='blue', density=True)
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid(True)
        plt.savefig(os.path.join(save_path, filename))
        plt.close()

    plot_and_save(backdoored_weights, 'Backdoored Model Weights Distribution', 'backdoored_weights_hist.png')
    plot_and_save(target_weights, 'Target Model Weights Distribution', 'target_weights_hist.png')
    if backdoored_biases.numel() > 0:
        plot_and_save(backdoored_biases, 'Backdoored Model Biases Distribution', 'backdoored_biases_hist.png')
    if target_biases.numel() > 0:
        plot_and_save(target_biases, 'Target Model Biases Distribution', 'target_biases_hist.png')