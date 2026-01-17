import torch

def collate_fn_motion(batch):
    """
    Custom collate function to pad motion to the max length in the batch and create an attention mask.

    Params
    -------
    batch : list[tensor]
        List of motions in the batch.
    
    Returns
    -------
    padded_batch : tensor
        Padded batch of motions.
    attn_mask : tensor
        Attention mask indicating the valid positions in the sequences.
    """
    lengths = [motion.shape[0] for motion in batch]
    max_length = max(lengths)

    padded = torch.zeros(len(batch), max_length, 22, 3)  # Assuming motion has shape (T, 22, 3)
    attn_mask = torch.zeros(len(batch), max_length)

    for i, motion in enumerate(batch):
        length = motion.shape[0]
        padded[i, :length] = motion
        attn_mask[i, :length] = 1   

    return padded, attn_mask