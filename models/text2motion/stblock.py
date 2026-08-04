# This code implementation is inspired from the following paper 
# "Scalable Diffusion Models with Transformers"
# ------------------------------------------------------------------------------
# Paper: https://arxiv.org/pdf/2212.09748
# Code: https://github.com/facebookresearch/DiT/blob/main/models.py
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from ..motion2text.transformers.blocks import PositionalEmbedding, FeedForward
from ..motion2text.graph.blocks import SpatioTemporalAttention
from ..motion2text.graph.decoders import TDecoder
from typing import Optional

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class StTBlock(nn.Module):
    """
    A spatio-temporal transformer model for motion denoising, conditioned on text embeddings.
    """

    def __init__(self,
                 model_dim: int = 512, 
                 ff_dim: int = 512,
                 num_heads: int = 8, 
                 dropout: float = 0.2,
                 swiglu: bool = False):
        """
        Params
        -------
        motion_dim: int
            Dimensionality of the input motion data.
        model_dim: int
            The dimensionality of the model's embeddings.
        ff_dim: int
            Dimensionality of the inner layer in the feed-forward network.
        num_heads: int
            Number of attention heads in the multi-head attention mechanism.
        dropout: float
            Dropout rate for regularization. Defaults to 0.2.
        swiglu: bool
            Whether to use SwiGLU activation function. Defaults to False.
        """
        super(StTBlock, self).__init__()

        self.layer_norm_1 = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.s_atten = SpatioTemporalAttention(model_dim, num_heads, dropout)
        self.layer_norm_2 = nn.LayerNorm(model_dim, elementwise_affine=False)

        mlp_hidden_dim = ff_dim
        self.mlp = FeedForward(model_dim, model_dim, mlp_hidden_dim, swiglu=swiglu)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, model_dim * 6, bias=True))

    def forward(self,
                x: torch.Tensor,
                c: torch.Tensor):
        """
        Run encoder-decoder to predict description sequence.

        Parameters
        ----------
        x : torch.Tensor
            Embedded input motion sequences
        c : torch.Tensor
            Embedded conditioning information (timestep + text embeddings)

        Returns
        -------
        torch.Tensor
            Embedded output motion sequences.
        """
        gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2 = self.adaLN(c).chunk(6, dim=1)

        x_norm_1 = self.layer_norm_1(x)
        x_modulated_1 = modulate(x_norm_1, beta_1, gamma_1)
        res_con_1 = x + alpha_1 * self.s_atten(x_modulated_1, None)

        x_norm_2 = self.layer_norm_2(res_con_1)
        x_modulated_2 = modulate(x_norm_2, beta_2, gamma_2)
        res_con_2 = res_con_1 + alpha_2 * self.mlp(x_modulated_2)

        return res_con_2
