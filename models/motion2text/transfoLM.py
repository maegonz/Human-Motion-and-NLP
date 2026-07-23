import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from .transformers.blocks import PositionalEmbedding, FeedForward
from .transformers.encoders import Encoder
from .graph.encoders import STEncoder, TEncoder
from .graph.blocks import JointAggregator
from typing import Optional

class TransfoLM(nn.Module):
    def __init__(self,
                 motion_dim: int,
                 lm_name: str = 't5-small',
                 model_dim: int = 512, 
                 num_heads: int = 8, 
                 num_layers: int = 6, 
                 ff_dim: int = 512,
                 max_seq_len: int = 100,
                 dropout: float = 0.2,
                 kernel_size: Optional[int] = 3,
                 graph: bool = True):
        """
        Params
        -------
        motion_dim: int
            Dimensionality of the input motion data.
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
        
        super(TransfoLM, self).__init__()
        self.max_seq_len = max_seq_len

        # Embedding layers
        # Consider using a more complex embedding strategy for motion data
        self.enc_embedding = FeedForward(motion_dim, model_dim, model_dim, dropout)
        self.pos_embedding = PositionalEmbedding(model_dim)
        self.dropout = nn.Dropout(dropout)

        # Encoder layers
        if graph:
            assert kernel_size is not None and kernel_size > 1, "Kernel size must be greater than 1 for temporal convolution."
            encoder_layers = (
                [STEncoder(model_dim, num_heads, kernel_size, dropout) for _ in range(num_layers)] + 
                [TEncoder(model_dim, num_heads, dropout, ff_dim) for _ in range(num_layers)]
            )
        else:
            encoder_layers = [
                Encoder(model_dim, num_heads, dropout, ff_dim) for _ in range(num_layers)
            ]

        self.encoder = nn.Sequential(*encoder_layers)
        
        # T5 decoder layers
        self.lm = T5ForConditionalGeneration.from_pretrained(lm_name)
        self.lm.eval()
        self.lm.config.use_cache = False    # Disable caching for training

        for param in self.lm.parameters():
            param.requires_grad = False     # Freeze LM parameters

        self.aggregator = JointAggregator(model_dim=model_dim, num_queries=4, num_heads=4, n_layers=3, dropout=dropout)

    
    def forward(self, 
                src: torch.Tensor, 
                tgt: Optional[torch.Tensor]=None, 
                encoder_attn_mask: Optional[torch.Tensor]=None, 
                generation: bool=False, 
                prefix_ids: Optional[torch.Tensor]=None):
        
        B, T, J, _ = src.shape                 # src: (B, T, J, 3)
        device = src.device

        # Motion Embedding          
        src_emb = self.enc_embedding(src)        # (B, T, J, model_dim)
        src_emb = self.pos_embedding(src_emb)
        src_emb = self.dropout(src_emb)

        # Spatial-Temporal encoder
        encoder_output = src_emb
        for layer in self.encoder:
            encoder_output = layer(encoder_output, mask=encoder_attn_mask)

        # TODO: Ensure that the attention mask is correctly updated to reflect the concatenated sequence length
        if encoder_attn_mask is None:
            # If no mask is given, assume all incoming motion frames are valid (shape: B, T, J)
            encoder_attn_mask = torch.ones(B, T, J, device=src.device, dtype=torch.float32)
        else:
            encoder_attn_mask = encoder_attn_mask.to(dtype=torch.float32)

        # Apply joint aggregation to get a single representation per frame
        motion_embeds, encoder_attn_mask = self.aggregator(encoder_output, mask=encoder_attn_mask)  # (B, T, model_dim)

        # -- Preparation for T5 sequence --
        if prefix_ids is not None:
            prefix_len = prefix_ids.shape[1]
            if prefix_ids.device != device:
                prefix_ids = prefix_ids.to(device, non_blocking=True) 

            if prefix_ids.size(0) != B:
                # Expand prefix_ids to match the batch size
                prefix_ids = prefix_ids.expand(B, -1)

            # Get T5's internal embeddings for the text prefix
            prefix_embeds = self.lm.shared(prefix_ids)  # (B, prefix_len, model_dim)
            prefix_mask = torch.ones(B, prefix_len, device=device, dtype=torch.float32)
        
            combined_embeds = torch.cat([prefix_embeds, motion_embeds], dim=1)
            combined_mask = torch.cat([prefix_mask, encoder_attn_mask], dim=1)   # (B, prefix_len + T)
        else:
            combined_embeds = motion_embeds
            combined_mask = encoder_attn_mask  # (B, T)

        # Wrap for T5 compatibility
        t5_encoder_outputs = BaseModelOutput(last_hidden_state=combined_embeds)

        # === Generation Mode ===
        if generation:
            outputs_ids = self.lm.generate(
                encoder_outputs=t5_encoder_outputs,
                attention_mask=combined_mask,
                max_length=40,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,  
                repetition_penalty=2.0,  
                length_penalty=1.0
            )             
            return outputs_ids
        
        # === Training Mode ===
        # if tgt.device != device:
        #     tgt = tgt.to(device, non_blocking=True)

        # Forward pass
        outputs = self.lm(
            encoder_outputs=t5_encoder_outputs,
            attention_mask=combined_mask,
            labels=tgt,
            return_dict=True
        )

        # --- Masked Mean for Motion Embeddings ---
        motion_weights = encoder_attn_mask.unsqueeze(-1) # (B, T, 1)
        motion_sum_len = motion_weights.sum(dim=1).clamp_min(1.0)  # Avoid division by zero
        # Zero out padding frames, sum them up, and divide by actual sequence lengths
        motion_embeddings = (motion_embeds * motion_weights).sum(dim=1) / motion_sum_len  # (B, model_dim)

        # --- Masked Mean for Text Embeddings ---
        tgt_weights = (tgt != 0).float().unsqueeze(-1).to(dtype=torch.float32)
        text_embeds_raw = self.lm.shared(tgt)
        text_sum_len = tgt_weights.sum(dim=1).clamp_min(1.0)  # Avoid division by zero
        text_embeddings = (text_embeds_raw * tgt_weights).sum(dim=1) / text_sum_len
                
        return {
            "outputs": outputs,
            "motion_embeddings": motion_embeddings,
            "text_embeddings": text_embeddings,
            "loss": outputs.loss,
            "logits": outputs.logits
        }