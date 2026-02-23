import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from ..metrics import scores, contrastive_loss


def training(model: nn.Module,
            train_loader: DataLoader,
            criterion: nn.Module,
            optimizer: Optimizer,
            device: torch.device,
            epochs: int,
            val_loader: DataLoader=None,
            alpha: float=0.1,
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
    alpha : float, optional
        Weighting factor for the contrastive loss. Default is 0.1.
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
    use_amp = use_amp and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    train_losses = []
    val_losses = []

    epoch_tqdm = tqdm(range(epochs), desc="Training Progress")

    for epoch in epoch_tqdm:
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for item in loop:
            motion = item['motion'].to(device, non_blocking=True)
            captions_tokens = item['input_ids'].squeeze(1).to(device, non_blocking=True)
            t5_attn_mask = item['t5_attn_mask'].squeeze(1).to(device, non_blocking=True)
            encoder_attn_mask = item['attn_mask'].squeeze(1).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(motion, captions_tokens, encoder_attn_mask=encoder_attn_mask, t5_attn_mask=t5_attn_mask)

                # logits = outputs["logits"] # (batch_size, seq_len, vocab_size)
                # print("logits.shape:", logits.shape)
                # logits = logits.view(-1, logits.size(-1))  # (batch * seq_len, vocab_size)
                # print("logits reshaped to:", logits.shape)
                # # Ensure labels are the ground truth shifted correctly
                # labels = captions_tokens.clone()
                # print("labels.shape before shift:", labels.shape)
                # labels[labels == model.tokenizer.pad_token_id] = -100 
                # labels = labels.view(-1) # (batch * seq_len)
                # print("labels reshaped to:", labels.shape)
                # ce_loss = criterion(logits, labels)

                motion_features, text_features = outputs["motion_embeddings"], outputs["text_embeddings"]
                
                ce_loss = outputs["loss"]
                cl_loss = contrastive_loss(motion_features, text_features)

                # Combine them (lambda is a hyperparameter, start with 0.1)
                loss = ce_loss + (alpha * cl_loss)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * motion.size(0)
            # _, predicted = torch.max(outputs, 1)
            loop.set_postfix(loss=loss.item())

            del outputs, loss, motion, captions_tokens, t5_attn_mask, encoder_attn_mask, motion_features, text_features

        torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        if val_loader is not None:
            val_loss, _ = evaluation(
                model, val_loader, device, use_amp
            )
            val_losses.append(val_loss)

            epoch_tqdm.set_postfix(
                train_loss=epoch_loss,
                val_loss=val_loss,
            )
        else:
            epoch_tqdm.set_postfix(train_loss=epoch_loss)

    return train_losses, val_losses

        
def evaluation(model: nn.Module, 
               data_loader: DataLoader,
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
    metrics_scores = {'bleu': [], 'meteor': [], 'rouge1': [], 'rouge2': [], 'rougeL': [], 'cider': []}

    with torch.no_grad():
        for id, item in enumerate(data_loader):
            motion = item['motion'].to(device, non_blocking=True)
            captions_tokens = item['input_ids'].squeeze(1).to(device, non_blocking=True)
            t5_attn_mask = item['t5_attn_mask'].squeeze(1).to(device, non_blocking=True)
            encoder_attn_mask = item['attn_mask'].squeeze(1).to(device, non_blocking=True)
            captions = item['captions']

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(motion, captions_tokens, encoder_attn_mask=encoder_attn_mask, t5_attn_mask=t5_attn_mask, generation=True)

            batch_metrics = {k: [] for k in metrics_scores.keys()}

            for caption, output in zip(captions, outputs):
                output = output.split("describe the motion in English: ")[-1].strip()
                metrics = scores(caption, output)
                for key in metrics_scores.keys():
                    batch_metrics[key].append(metrics[key])

            for key in metrics_scores.keys():
                if batch_metrics[key]:  # Check if the list is not empty
                    batch_mean = sum(batch_metrics[key]) / len(batch_metrics[key])
                else:
                    batch_mean = 0.0  # or some default value if there are no metrics
                
                metrics_scores[key].append(batch_mean)

    return metrics_scores