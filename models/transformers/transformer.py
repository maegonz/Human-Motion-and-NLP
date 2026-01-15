import torch
import torch.nn as nn
from .blocks import PositionalEmbedding
from .encoders import Encoder
from .decoders import Decoder

class Transformer(nn.Module):
    def __init__(self,
                 motion_dim: int,
                 tgt_vocab_size: int,
                 model_dim: int = 512, 
                 num_heads: int = 8, 
                 num_layers: int = 6, 
                 ff_dim: int = 2048,
                 max_seq_len: int = 100,
                 dropout: float=0.2):
        """
        Params
        -------
        motion_dim: int
            Dimensionality of the input motion data.
        tgt_vocab_size: int
            Target vocabulary size.
        model_dim: int
            The dimensionality of the model's embeddings.
        num_heads: int
            Number of attention heads in the multi-head attention mechanism.
        num_layers: int
            Number of layers for both the encoder and the decoder.
        ff_dim: int
            Dimensionality of the inner layer in the feed-forward network.
        max_seq_len: int
            Maximum sequence length for positional encoding.
        dropout: float
            Dropout rate for regularization. Defaults to 0.2.
        """
        super(Transformer, self).__init__()
        # Embedding layers
        self.enc_embedding = nn.Linear(motion_dim, model_dim)
        self.deco_embedding = nn.Embedding(tgt_vocab_size, model_dim)
        self.pos_embedding = PositionalEmbedding(model_dim, max_seq_len)
        
        # Encoder layers
        self.encoder = nn.ModuleList(
            [Encoder(model_dim, num_heads, dropout, ff_dim) for _ in range(num_layers)]
        )

        # Decoder layers
        self.decoder = nn.ModuleList(
            [Decoder(model_dim, num_heads, dropout, ff_dim) for _ in range(num_layers)]
        )

        # Projection layer to map decoder output to target vocabulary size
        self.fc = nn.Linear(model_dim, tgt_vocab_size)
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
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, src_seq_len)

        # Target mask
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, tgt_seq_len)
        tgt_sub_mask = torch.tril(torch.ones((seq_len, seq_len))).bool()  # (tgt_seq_len, tgt_seq_len)
        tgt_mask = tgt_pad_mask & tgt_sub_mask  # (batch_size, 1, tgt_seq_len, tgt_seq_len)

        return src_mask, tgt_mask


    def forward(self, src, tgt):
        """
        Run encoder-decoder to predict description sequence.

        Parameters
        ----------
        src_rec : torch.Tensor
           
        tgt : torch.Tensor
            Target description sequence 

        Returns
        -------
        torch.Tensor
            Predicted description sequence of shape (batch, tgt_vocab_size).
        """
        src_mask, tgt_mask = self.generate_mask(src, tgt)

        # Embedding and positional encoding
        # src: (B, T, 22, 3)
        B, T, _, _ = src.shape

        src = src.view(B, T, 66)           # flatten joints
        src_emb = self.enc_embedding(src)  # (B, T, model_dim)
        src_emb = self.pos_embedding(src_emb)
        src_emb = self.dropout(src_emb)

        tgt_emb = self.deco_embedding(tgt)
        tgt_emb = self.pos_embedding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        # Encoder
        encoder_output = src_emb
        for encoder in self.encoder:
            encoder_output = encoder(encoder_output, src_mask)

        # Decoder
        decoder_output = tgt_emb
        for decoder in self.decoder:
            decoder_output = decoder(decoder_output, encoder_output, src_mask, tgt_mask)

        # Final linear layer
        description_output = self.fc(decoder_output)

        return description_output