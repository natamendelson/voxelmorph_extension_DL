import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Helper functions.
def normalize(x):
  x -= x.min()
  return x / x.max()


def show(im, title=None):
    im = np.asarray(im).reshape((256, 256, 256))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    slices = im[:, 125, :], im[:, :, 127], im[127, :, :].T
    for im_slice, ax in zip(slices, axes):
        ax.imshow(im_slice, cmap='gray')
        ax.set_axis_off()

    if title:
        axes[1].text(0.50, 1.05, title, ha='center', transform=axes[1].transAxes, size=14)

def mse(fixed, moving, warped, name):
    # warped_fixed_axes = warped.transpose(2, 1, 0)
    if "SSIM" not in name:
        warped = normalize(warped)
    #normlize other two images:
    fixed = normalize(fixed)
    moving = normalize(moving)

    error_after_reg = np.mean((fixed - warped)**2)
    return error_after_reg

def load_volume(path):
    return np.load(path)['vol']


def visualize_feature_channel(features, channel_index=0):
    feature_map = features[channel_index] 
    num_slices = feature_map.shape[0]

    fig, axes = plt.subplots(8, 8, figsize=(12, 6))  
    axes = axes.flatten()

    for i in range(num_slices):
        ax = axes[i]
        im = ax.imshow(feature_map[i][0])
 
  # Show slice i as heatmap
        ax.set_title(f'Slice {i}')
        ax.axis('off')

    fig.suptitle(f'Feature Channel {channel_index} (slices)')
    plt.tight_layout()
    plt.show()

def save_normalized(input_path):
    vol = load_volume(input_path)
    vol_norm = normalize(vol)
    input_path = Path(input_path)
    output_dir = input_path.parent / "normalized"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / input_path.name
    np.savez_compressed(output_path, vol=vol_norm)

def plot_loss_curves(file_paths, labels, title='Training Loss Over Epochs', xlabel='Epoch', ylabel='Loss'):
    all_losses = []

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            losses = [float(line.strip()) for line in f if line.strip()]
            all_losses.append(losses)

    plt.figure(figsize=(12, 8))
    for losses, label in zip(all_losses, labels):
        plt.plot(losses, label=label)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([0,23])
    plt.legend()
    plt.figure(figsize=(12, 6))
    plt.show()