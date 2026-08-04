import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from .unet import SimpleUNet
from .methods import closed_form_sampling
from .stdiffusion import StTDiffusion

def linear_beta_schedule(timesteps: int, start: float=0.0001, end: float=0.02):
    """Linear noise schedule for the diffusion process"""
    return torch.linspace(start, end, timesteps)

# Let's test the model
def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = StTDiffusion().to(device)

    batch_size = 4

    # Dummy inputs
    x = torch.randn(batch_size, 3, 64, 64).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    text_indices = torch.randint(0, 10, (batch_size,)).to(device)


    # Forward pass
    with torch.no_grad():
        predicted_noise = model(x, t, text_indices)

    print(f"Noisy image shape: {x.shape}")
    print(f"Text indices shape: {text_indices.shape}")
    print(f"Predicted noise shape: {predicted_noise.shape}")
    print("✓ Model forward pass successful!")


# Let's test the forward diffusion process
def visualize_forward_process_image(image_path):
    """Visualize the forward diffusion process on a sample image"""
    # Load a sample image
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])

    # Use a sample image (you can replace this with any image)
    sample_image = Image.open(image_path)
    sample_image = transforms.ToTensor()(sample_image)

    # Set up diffusion parameters
    timesteps = 1000
    betas = linear_beta_schedule(timesteps)

    # Select specific timesteps to visualize
    viz_timesteps = [0, 50, 100, 200, 500, 999]

    fig, axes = plt.subplots(1, len(viz_timesteps), figsize=(15, 3))

    for i, t in enumerate(viz_timesteps):
        x_t, epsilon = closed_form_sampling(sample_image, torch.tensor([t]), betas)

        # Denormalize for visualization
        img = x_t[0].permute(1, 2, 0)
        # img = (img * 0.5 + 0.5).clamp(0, 1)

        axes[i].imshow(img)
        axes[i].set_title(f't = {t}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()