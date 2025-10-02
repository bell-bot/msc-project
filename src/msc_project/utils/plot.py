import os
import matplotlib.pyplot as plt

def plot_histograms(backdoored_weights, target_weights, backdoored_biases, target_biases, save_path):

    os.makedirs(save_path, exist_ok=True)

    def plot_and_save(data, title, filename):
        plt.figure(figsize=(10, 6))
        plt.hist(data.numpy(), bins=100, alpha=0.7, color='blue', density=True)
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.yscale("log")
        plt.grid(True)
        plt.savefig(os.path.join(save_path, filename))
        plt.close()

    plot_and_save(backdoored_weights, 'Backdoored Model Weights Distribution', 'backdoored_weights_hist.png')
    plot_and_save(target_weights, 'Target Model Weights Distribution', 'target_weights_hist.png')
    if backdoored_biases.numel() > 0:
        plot_and_save(backdoored_biases, 'Backdoored Model Biases Distribution', 'backdoored_biases_hist.png')
    if target_biases.numel() > 0:
        plot_and_save(target_biases, 'Target Model Biases Distribution', 'target_biases_hist.png')