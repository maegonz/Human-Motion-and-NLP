import torch
import torch.nn as nn
from .blocks import SpatioTemporalAttention, TemporalConv


class STEncoder(nn.Module):
    """
    Spatial-Temporal Transformer Encoder for motion data.
    """
    def __init__(self, model_dim: int, num_heads: int, kernel_size: int=3, dropout: float=0.2):
        super(STEncoder, self).__init__()
        self.s_atten = SpatioTemporalAttention(model_dim, num_heads, dropout, mode="spatial")
        self.layer_norm = nn.LayerNorm(model_dim)
        self.temporal_conv = TemporalConv(model_dim, kernel_size, dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask):
        s_atten_ouptut = self.s_atten(x, x, x, mask)
        temporal_conv = self.temporal_conv(s_atten_ouptut)
        encoder_output = self.layer_norm(x + self.dropout(temporal_conv))
        return encoder_output
    

from models.motion2text.transformers.blocks import FeedForward

class TEncoder(nn.Module):
    """
    Temporal Transformer Encoder for motion data.
    """
    def __init__(self, model_dim: int, num_heads: int, dropout: float, ff_dim: int):
        super(TEncoder, self).__init__()
        self.t_atten = SpatioTemporalAttention(model_dim, num_heads, dropout, mode="temporal")
        self.layer_norm = nn.LayerNorm(model_dim)
        self.ff = FeedForward(model_dim, ff_dim, dropout)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask):
        t_atten_ouptut = self.t_atten(x, x, x, mask)
        res_norm = self.layer_norm(x + self.dropout(t_atten_ouptut))
        feed_forward = self.ff(res_norm)
        encoder_output = self.layer_norm(res_norm + self.dropout(feed_forward))
        return encoder_output