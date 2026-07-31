import torch
import torch.nn as nn
from .transformers.blocks import PositionalEmbedding, FeedForward, FastMultiHeadAttention
from .transformers.decoders import Decoder
from .graph.encoders import STEncoder, TEncoder
from .graph.decoders import TDecoder
from .graph.blocks import JointAggregator
from ..utils import global_mean_pooling
from typing import Optional

class STformer(nn.Module):
    def __init__(self,
                 motion_dim: int,
                 tgt_vocab_size: int,
                 model_dim: int = 512, 
                 ff_dim: int = 512,
                 num_heads: int = 8, 
                 num_layers: int = 6,
                 kernel_size: int = 3, 
                 max_seq_len: int = 100,
                 dropout: float = 0.2,
                 swiglu: bool = False):
        """
        Params
        -------
        motion_dim: int
            Dimensionality of the input motion data.
        tgt_vocab_size: int
            Target vocabulary size.
        model_dim: int
            The dimensionality of the model's embeddings.
        ff_dim: int
            Dimensionality of the inner layer in the feed-forward network.
        num_heads: int
            Number of attention heads in the multi-head attention mechanism.
        num_layers: int
            Number of layers for both the encoder and the decoder.
        kernel_size: int
            Kernel size for the spatial-temporal encoder.
        max_seq_len: int
            Maximum sequence length for positional encoding.
        dropout: float
            Dropout rate for regularization. Defaults to 0.2.
        swiglu: bool
            Whether to use SwiGLU activation function. Defaults to False.
        """
        super(STformer, self).__init__()
        # Embedding layers
        self.enc_embedding = FeedForward(motion_dim, model_dim, model_dim)
        self.deco_embedding = nn.Embedding(tgt_vocab_size, model_dim)
        self.pos_embedding = PositionalEmbedding(model_dim)
        
        # Encoder layers
        self.encoder = nn.ModuleList(
            [STEncoder(model_dim, num_heads, kernel_size, dropout) for _ in range(num_layers)] + 
            [TEncoder(model_dim, num_heads, dropout, ff_dim) for _ in range(num_layers)]
        )

        # Decoder layers
        self.decoder = nn.ModuleList(
            [TDecoder(model_dim, num_heads, dropout, ff_dim, swiglu=swiglu) for _ in range(num_layers)]
        )

        self.aggregator = JointAggregator(model_dim=model_dim, num_queries=4, num_heads=4, n_layers=3, dropout=dropout)
        
        # Projection layer to map decoder output to target vocabulary size
        self.projection = FeedForward(model_dim, tgt_vocab_size, tgt_vocab_size, swiglu=swiglu)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        """
        Params
        -------
        src: torch.Tensor
            input to the encoder
        tgt: torch.Tensor
            input to the decoder

        Returns
        -------
        returns source and target masks
        """
        seq_len = tgt.size(1)

        # Source mask
        src_mask = (src != 0)  # (batch_size, src_seq_len)

        # Target mask
        tgt_pad_mask = (tgt != 0)  # (batch_size, tgt_seq_len)
        # tgt_sub_mask = torch.tril(torch.ones((seq_len, seq_len), device=tgt.device)).bool()  # (tgt_seq_len, tgt_seq_len)
        # tgt_mask = tgt_pad_mask & tgt_sub_mask  # (batch_size, 1, tgt_seq_len, tgt_seq_len)
        tgt_mask = tgt_pad_mask

        return src_mask, tgt_mask


    def forward(self,
                src: torch.Tensor, 
                tgt: Optional[torch.Tensor]=None, 
                encoder_attn_mask: Optional[torch.Tensor]=None, 
                prefix_ids: Optional[torch.Tensor]=None):
        """
        Run encoder-decoder to predict description sequence.

        Parameters
        ----------
        src : torch.Tensor
           
        tgt : torch.Tensor
            Target description sequence 

        Returns
        -------
        torch.Tensor
            Predicted description sequence of shape (batch, tgt_vocab_size).
        """
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_mask = encoder_attn_mask if encoder_attn_mask is not None else src_mask

        # Embedding and positional encoding
        # src: (B, T, 22, 3)
        B, T, J, _ = src.shape
        device = src.device

        src_emb = self.enc_embedding(src)     # (B, T, J, model_dim)
        src_emb = self.pos_embedding(src_emb)
        src_emb = self.dropout(src_emb)

        tgt_emb = self.deco_embedding(tgt)
        tgt_emb = self.pos_embedding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        # Encoder
        encoder_output = src_emb
        for encoder in self.encoder:
            encoder_output = encoder(encoder_output, src_mask)

        # Apply joint aggregation to get a single representation per frame
        # encoder_output, src_mask = self.aggregator(encoder_output, mask=src_mask)

        # Decoder
        decoder_output = tgt_emb
        for decoder in self.decoder:
            decoder_output = decoder(decoder_output, encoder_output, src_mask, tgt_mask)

        # Final linear layer
        description_output = self.projection(decoder_output).squeeze(2)  # [B, T, vocab_size]

        # =====================================================
        #  Global Mean Pooling for motion and text embeddings
        # =====================================================
        # --- Masked Mean for Motion Embeddings ---
        motion_embeddings = encoder_output.view(B, T * J, -1)  # (B, T*J, model_dim)
        mask_embedddings = src_mask.unsqueeze(-1).expand(-1, -1, J).reshape(B, -1).contiguous()  # (B, T*J)
        motion_embeddings = global_mean_pooling(motion_embeddings, mask=mask_embedddings)  # (B, model_dim)
        # --- Masked Mean for Text Embeddings ---
        text_embeddings = tgt_emb.squeeze(2)  # (B, T, model_dim)
        text_embeddings = global_mean_pooling(text_embeddings, mask=tgt_mask)  # (B, model_dim)

        return {
            "decoder_output": description_output,
            "motion_embeddings": motion_embeddings,
            "text_embeddings": text_embeddings,
            "loss": None
        }