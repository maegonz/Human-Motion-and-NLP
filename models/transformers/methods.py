import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from metrics import score


def training(model: nn.Module,
            train_loader: DataLoader,
            criterion: nn.Module,
            optimizer: Optimizer,
            device: torch.device,
            epochs: int,
            val_loader: DataLoader=None,
            use_amp: bool=True):
    """
    Train a PyTorch model with optional Automatic Mixed Precision.

    Parameters
    ----------
    model : nn.Module
        The neural network model to be trained.
    train_loader : DataLoader
        DataLoader providing the training dataset.
    criterion : nn.Module
        Loss function used to compute training loss.
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    device : torch.device
        Device on which to train the model ('cpu' or 'cuda').
    epochs : int
        Number of training epochs.
    val_loader : DataLoader, optional
        DataLoader providing the validation dataset.
        If None, no validation is performed. Default is None.
    use_amp : bool, optional
        Whether to use AMP.
        AMP is enabled only when using a CUDA device. Default is True.

    Returns
    -------
    train_losses : list of float
        Average training loss for each epoch.
    train_accuracies : list of float
        Training accuracy (percentage) for each epoch.
    val_losses : list of float
        Validation loss for each epoch.
        Empty if val_loader is None.
    val_accuracies : list of float
        Validation accuracy (percentage) for each epoch.
        Empty if val_loader is None.
    """

    model.to(device)
    # scaler = GradScaler(enabled=(use_amp and device.type == "cuda"))
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    epoch_tqdm = tqdm(range(epochs), desc="Training Progress")

    for epoch in epoch_tqdm:
        model.train()
        running_loss = 0.0
        bleu_score = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for item in loop:
            motion, captions_tokens = item['motion'].to(device), item['input_ids'][0].to(device)
            optimizer.zero_grad()

            with autocast(device_type=device.type, enabled=use_amp):
                decoder_input = captions_tokens[:, :-1]
                labels = captions_tokens[:, 1:]

                outputs = model(motion, decoder_input)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * motion.size(0)
            _, predicted = torch.max(outputs, 1)
            bleu_score += score(captions_tokens, outputs)

            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = 100.0 * bleu_score / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        if val_loader is not None:
            val_loss, val_acc = evaluating(
                model, val_loader, criterion, device, use_amp
            )
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

            epoch_tqdm.set_postfix(
                train_loss=epoch_loss,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )
        else:
            epoch_tqdm.set_postfix(train_loss=epoch_loss)

    return train_losses, train_accuracies, val_losses, val_accuracies

        
def evaluating(model: nn.Module, 
               data_loader: DataLoader,
               criterion: nn.Module,
               device: torch.device,
               use_amp: bool = True):
    """
    Evaluate a PyTorch model on a dataset with optional AMP.

    Parameters
    ----------
    model : nn.Module
        The trained model to be evaluated.
    data_loader : DataLoader
        DataLoader providing the evaluation dataset.
    criterion : nn.Module
        Loss function used to compute evaluation loss.
    device : torch.device
        Device on which evaluation is performed ('cpu' or 'cuda').
    use_amp : bool, optional
        Whether to use Automatic Mixed Precision (AMP).
        AMP is enabled only when using a CUDA device. Default is True.

    Returns
    -------
    avg_loss : float
        Average loss over the entire dataset.
    avg_accuracy : float
        Average accuracy (percentage) over the entire dataset.
    """

    model.eval()
    total_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    avg_accuracy = 100.0 * correct / len(data_loader.dataset)

    return avg_loss, avg_accuracy