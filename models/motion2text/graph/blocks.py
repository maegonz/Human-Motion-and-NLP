import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Union

class SpatioTemporalAttention(nn.Module):
    """
    Implementation of Spatial-Temporal Multi-Head Self-Attention sub-layer for spatio-temporal graphs,
    based on S2TNet paper https://arxiv.org/pdf/2206.10902

    This layer captures spatial interactions between different nodes (e.g., joints)
    independently at each frame in a temporal sequence. 
    It computes scaled dot-product attention across the spatial and the temporal dimension.

    Parameters
    ----------
    model_dim : int, default=256
        The total dimensionality of the input and output features (D).
    num_heads : int, default=4
        Number of attention heads (H). `model_dim` must be divisible 
        by `num_heads`.
    dropout : float, default=0.2
        Dropout probability applied to the attention scores during the 
        softmax operation.
    mode : str, default="spatial"
        Specifies the mode of attention. Currently supports "spatial" for
        spatial attention across joints and "temporal" for temporal attention.

    Attributes
    ----------
    model_dim : int
        Dimensionality of the model's hidden states.
    num_heads : int
        Number of parallel attention heads.
    head_dim : int
        Dimensionality of each individual attention head (D_head = model_dim // num_heads).
    dropout : torch.nn.Dropout
        Dropout layer for regularization.
    qkv_projection : torch.nn.Linear
        Linear layer to project the input into Query, Key, and Value spaces simultaneously.
    output : torch.nn.Linear
        Final linear projection layer applied after concatenating all attention heads.

    Raises
    ------
    AssertionError
        If `model_dim` is not divisible by `num_heads`.

    Notes
    -----
    This layer expects the Query, Key, and Value tensors to be identical for 
    Self-Attention processing. The spatial attention maps are computed 
    per time step, meaning interactions are localized within each intra-frame.

    Examples
    --------
    >>> batch_size, seq_len, num_joints, model_dim = 2, 6, 22, 256
    >>> spatial_attn = SpatioTemporalAttention(model_dim=model_dim, num_heads=4)
    >>> x = torch.randn(batch_size, seq_len, num_joints, model_dim)
    >>> # For self-attention, Q, K, and V must be the same tensor
    >>> out = spatial_attn(x, x, x)
    >>> out.shape
    torch.Size([2, 6, 22, 256])
    """
    def __init__(self, model_dim: int = 256, num_heads: int = 4, dropout: float=0.2, mode: str="spatial"):
        super(SpatioTemporalAttention, self).__init__()
        assert model_dim % num_heads == 0, "Model's dimension must be divisible by num_heads"
        assert mode in ["spatial", "temporal"], "Mode must be either 'spatial' or 'temporal'"

        self.mode = mode
        self.model_dim = model_dim
        self.num_heads = num_heads              # h
        self.head_dim = model_dim // num_heads  # d_k

        self.dropout = nn.Dropout(dropout)
        self.qkv_projection = nn.Linear(model_dim, model_dim * 3)  # 3 for Q, K, V
        self.output = nn.Linear(model_dim, model_dim)

    def sdp_attention(self, query, key, value, attn_mask=None, dropout_p=0.0, scale=None):
        """
        Compute scaled dot-product attention using PyTorch's optimized implementation.
        """

        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(self.head_dim) if scale is None else scale
        # [B, H, J, S, D_head] @ [B, H, J, D_head, S] -> [B, H, J, S, S]
        attn_weight = query @ key.transpose(-2, -1) * scale_factor

        if self.mode == "temporal":
            attn_weight = attn_weight.masked_fill(attn_mask, -1e9)

        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        output = attn_weight @ value  # [B, H, J, S, D_head]

        if self.mode == "spatial":
            output = output.masked_fill(attn_mask, float(0.0))
            
        return output

    def forward(self, Q, K, V, mask=None):
        assert Q.shape == K.shape == V.shape, "Q, K, and V must have the same shape"
        assert Q is K and K is V, "For Self-Attention, Q, K, and V should be the same"
        batch_size, seq_len, num_joints, _ = Q.size()
        
        # [B, S, J, D] -> [B, S, J, 3 * D]
        qkv = self.qkv_projection(Q)
        # -> [B, S, J, 3, H, D_head]
        qkv = qkv.view(batch_size, seq_len, num_joints, 3, self.num_heads, self.head_dim)
        
        if self.mode == "spatial":
            # -> [3, B, H, S, J, D_head]
            qkv = qkv.permute(3, 0, 4, 2, 1, 5)
        else:
            # -> [3, B, H, J, S, D_head]
            qkv = qkv.permute(3, 0, 4, 1, 2, 5)
        
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Boolean mask necessary (True for keeping, False for masking) or a float mask.
        if mask is not None:
            if  mask.dtype != torch.bool:
                mask = mask.to(torch.bool)
            if self.mode == "spatial":
                # [B, S] -> [B, 1, S, 1, 1]
                mask = mask[:, None, None, :, None]
            else:
                # [B, S] -> [B, 1, 1, 1, S]
                mask = mask[:, None, :, None, None]

        atten_output = self.sdp_attention(        # F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask,
            dropout_p=self.dropout.p,
        )

        if self.mode == "spatial":
            # [B, H, S, J, D_head] -> [B, S, J, H, D_head]
            atten_output = atten_output.permute(0, 2, 3, 1, 4).contiguous()
        else:
            # [B, H, J, S, D_head] -> [B, S, J, H, D_head]
            atten_output = atten_output.permute(0, 3, 2, 1, 4).contiguous()
    
        # [B, S, J, H, D_head] -> [B, S, J, H * D_head]
        atten_output = atten_output.view(batch_size, seq_len, num_joints, self.model_dim)

        return self.output(atten_output)
    

from models.utils import conv_init, bn_init

class TemporalConv(nn.Module):
    def __init__(self, model_dim: int = 256, kernel_size: int = 3, dropout: float = 0.2):
        super(TemporalConv, self).__init__()
        self.model_dim = model_dim
        self.kernel_size = kernel_size
        # self.dropout = nn.Dropout(dropout)
        self.padding = ((kernel_size - 1) // 2, 0)  # Padding for 'same' convolution along temporal dimension
        
        self.conv = nn.Conv2d(in_channels=model_dim, out_channels=model_dim, kernel_size=(kernel_size, 1), padding=self.padding)
        self.bnorm = nn.BatchNorm2d(model_dim)
        # self.dropout = nn.Dropout(dropout)

        conv_init(self.conv)
        bn_init(self.bnorm, 1)
    
    def forward(self, x):
        # x: [B, S, J, D] -> [B, D, S, J]
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = self.bnorm(x)
        # x = F.relu(x)
        # x = self.dropout(x)
        # [B, D, S, J] -> [B, S, J, D]
        return x.permute(0, 2, 3, 1)
