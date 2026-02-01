import torch

def collate_fn_motion(batch):
    """
    Custom collate function to pad motion to the max length in the batch and create an attention mask.

    Params
    -------
    batch : list[dict]
        List of motions in the batch.
    
    Returns
    -------
    padded_batch : tensor
        Padded batch of motions.
    attn_mask : tensor
        Attention mask indicating the valid positions in the sequences.
    """

    # Compute the maximum length in the batch
    lengths = [item['motion'].shape[0] for item in batch]
    max_length = max(lengths)

    padded = torch.zeros(len(batch), max_length, 22, 3)  # Assuming motion has shape (T, 22, 3)
    attn_mask = torch.zeros(len(batch), max_length)

    for i, item in enumerate(batch):
        motion = torch.from_numpy(item['motion'])  # shape: (T, 22, 3)
        length = motion.shape[0]
        padded[i, :length] = motion
        attn_mask[i, :length] = 1   

    # Concatenate captions texts
    caption_texts = [item['captions'] for item in batch]

    # Concatenate captions tokens
    caption_tokens = torch.stack(
        [item['input_ids'].squeeze(0) for item in batch], dim=0
    )
    
    # Concatenate t5 attention masks
    t5_attn_mask = torch.stack(
        [item['t5_attn_mask'].squeeze(0) for item in batch], dim=0
    )

    return {
        "motion": padded,
        "attn_mask": attn_mask,
        "input_ids": caption_tokens,
        "t5_attn_mask": t5_attn_mask,
        "captions": caption_texts
    }