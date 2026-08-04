import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from ..metrics import mpjpe


def training(model: nn.Module,
            train_loader: DataLoader,
            optimizer: Optimizer,
            device: torch.device,
            epochs: int=200,
            criterion: nn.Module=nn.MSELoss(),
            use_amp: bool=True):
    """
    Train a PyTorch model with optional Automatic Mixed Precision.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be trained.
    train_loader : DataLoader
        DataLoader providing the training dataset.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device on which to train the model ('cpu' or 'cuda').
    epochs : int
        Number of training epochs. Defaults to 200.
    criterion : nn.Module
        The loss function to be used for training. Defaults to Mean Squared Error Loss.
    use_amp : bool, optional
        Whether to use AMP.
        AMP is enabled only when using a CUDA device. Default is True.

    Returns
    -------
    train_losses : list of float
        Average training loss for each epoch.
    """

    model.to(device)
    model.train()
    use_amp = use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    train_losses = []

    epoch_tqdm = tqdm(range(epochs), desc="Training Progress")

    for _ in epoch_tqdm:
        running_loss = torch.tensor(0.0, device=device)

        for k, item in enumerate(train_loader):

            motion_0 = item['motion'].to(device, non_blocking=True)
            captions_tokens = item['input_ids'].squeeze(1).to(device, non_blocking=True)
            encoder_attn_mask = item['attn_mask'].squeeze(1).to(device, non_blocking=True)

            batch_size = motion_0.size(0)

            # Sample timesteps for each sample in the batch
            t = model.sample_timesteps(batch_size).to(device)
            # Noise scheduling parameters
            betas = model.betas

            with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
                # Add noise to the motion data (forward diffusion)
                motion_t, noise = closed_form_sampling(motion_0, t, betas, device=device)

                predicted_noise = model(motion_t, captions_tokens, t, encoder_attn_mask=encoder_attn_mask)

                loss = criterion(predicted_noise, noise)
                # matching_loss = mpjpe(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.detach() * batch_size

        epoch_loss = running_loss.item() / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        epoch_tqdm.set_postfix(train_loss=epoch_loss)

    return train_losses


def closed_form_sampling(x_0: torch.Tensor, t: torch.Tensor, betas: torch.Tensor, device: str="cuda") -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample from q(x_t | x_0) using the reparameterization trick

    Parameters
    ----------
    x_0: torch.Tensor
        Original motion sequence (batch_size, seq_len, num_joints, spatial_dim)
    t: torch.Tensor
        Timestep (batch_size,)
    betas: torch.Tensor
        Noise schedule (timesteps,)
    device: str, optional
        Device to run on. Default is "cuda".

    Returns
    -------
    x_t: torch.Tensor
        Noisy motion sequence at timestep t
    epsilon: torch.Tensor
        The noise that was added
    """
    # Extract sqrt(alpha_bar) and sqrt(1-alpha_bar) for timestep t
    t = t.to(device)
    sqrt_alphas_cumprod = torch.sqrt(1. - betas).cumprod(dim=0).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - sqrt_alphas_cumprod ** 2)

    # Gather the appropriate values for timestep t
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].to(device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].to(device)

    # Reshape for broadcasting
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None]
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None]

    # Sample noise
    epsilon = torch.randn_like(x_0).to(device)

    # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * epsilon
    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * epsilon

    return x_t, epsilon