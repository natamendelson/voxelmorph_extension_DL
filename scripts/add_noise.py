import torch

def add_gaussian_noise(image_tensor, mean=0.0, std=0.01):
    """
    Adds Gaussian noise to an image tensor.
    Args:
        image_tensor (torch.Tensor): Input tensor of shape (B, C, D, H, W) or (B, C, H, W)
        mean (float): Mean of the Gaussian noise
        std (float): Standard deviation of the Gaussian noise
    Returns:
        torch.Tensor: Noisy image tensor
    """
    noise = torch.randn_like(image_tensor) * std + mean
    noisy_image = image_tensor + noise
    # Optionally clamp to maintain valid intensity range, e.g., [0, 1]
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    return noisy_image
