import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from .transformers.blocks import PositionalEmbedding, MLP
from .transformers.encoders import Encoder
from.graph.encoders import STEncoder, TEncoder
from typing import Optional

class TransfoLM(nn.Module):
    def __init__(self,
                 motion_dim: int,
                 lm_name: str = 't5-small',
                 model_dim: int = 512, 
                 num_heads: int = 8, 
                 num_layers: int = 6, 
                 ff_dim: int = 2048,
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
        # Consider using a more complex embedding strategy for motion data, such as a small CNN or MLP, instead of a simple linear layer.
        self.enc_embedding = MLP(motion_dim, model_dim)
        self.pos_embedding = PositionalEmbedding(model_dim)

        # Encoder layers
        if graph:
            assert kernel_size > 1, "Kernel size must be greater than 1 for temporal convolution."
            self.encoder = nn.ModuleList(
                [STEncoder(model_dim, num_heads, kernel_size, dropout) for _ in range(num_layers)] + 
                [TEncoder(model_dim, num_heads, dropout, ff_dim) for _ in range(num_layers)]
            )
        else:
            self.encoder = nn.ModuleList(
                [Encoder(model_dim, num_heads, dropout, ff_dim) for _ in range(num_layers)]
            )
        
        # Decoder layers
        self.lm = T5ForConditionalGeneration.from_pretrained(lm_name)
        for param in self.lm.parameters():
            param.requires_grad = False     # Freeze LM parameters
        self.lm.eval()
        self.lm.config.use_cache = False    # Disable caching for training

        # Projection layer 
        # Map motion encoder output to LM model dimension
        self.projection = MLP(model_dim=model_dim, lm_model_dim=self.lm.config.d_model, dropout=dropout)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, src, tgt=None, encoder_attn_mask=None, t5_attn_mask=None, generation=False, prefix_ids=None):
        # Embedding and positional encoding
        B, T, _, _ = src.shape                 # src: (B, T, 22, 3)

        # src = src.view(B, T, -1)               # flatten joints (dynamically handles the 66)
        src_emb = self.enc_embedding(src)      # (B, T, model_dim)
        src_emb = self.pos_embedding(src_emb)
        src_emb = self.dropout(src_emb)

        # Motion encoder
        encoder_output = src_emb
        for encoder in self.encoder:
            encoder_output = encoder(encoder_output, mask=encoder_attn_mask)

        # Project motion encoder output to LM model dimension
        motion_embeds = self.projection(encoder_output)  # (B, T, lm_d_model)

        # Expand prefix
        prefix_ids = prefix_ids.to(src.device).expand(B, -1)
        prefix_len = prefix_ids.shape[1]  
        
        # Get T5's internal embeddings for the text prefix
        # We use shared() to access the input embedding matrix of T5
        prefix_embeds = self.lm.shared(prefix_ids) 
        
        # Concatenate text prompt + motion sequence
        combined_embeds = torch.cat([prefix_embeds, motion_embeds], dim=1)
        
        # Update the attention mask to account for the new prefix length
        # TODO: Ensure that the attention mask is correctly updated to reflect the concatenated sequence length
        if encoder_attn_mask is None:
            # If no mask is given, assume all incoming motion frames are valid (shape: B, T)
            encoder_attn_mask = torch.ones(B, T, device=src.device, dtype=torch.float32)
        else:
            encoder_attn_mask = encoder_attn_mask.to(dtype=torch.float32)
        
        prefix_mask = torch.ones(B, prefix_len, device=src.device, dtype=torch.float32)  # Prefix text mask
        combined_mask = torch.cat([prefix_mask, encoder_attn_mask], dim=1)               # (B, prefix_len + T)

        # Wrap for T5 compatibility
        t5_encoder_outputs = BaseModelOutput(last_hidden_state=combined_embeds)

        if generation:
            outputs_ids = self.lm.generate(
                encoder_outputs=t5_encoder_outputs,
                attention_mask=combined_mask, # Pass the updated mask
                max_length=40,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,  
                repetition_penalty=2.0,  
                length_penalty=1.0
            )             
            # outputs = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True) 
            return outputs_ids
        

        tgt = tgt.long()
        labels = tgt.clone()

        # Forward pass
        outputs = self.lm(
            encoder_outputs=t5_encoder_outputs,
            attention_mask=combined_mask,
            labels=labels,
            return_dict=True
        )
        assert torch.isnan(outputs.logits).any().item() == False, "Logits contain NaN values!"
        assert torch.isinf(outputs.logits).any().item() == False, "Logits contain Inf values!"
        assert torch.isnan(outputs.loss).any().item() == False, "Loss contains NaN values!"
        assert torch.isinf(outputs.loss).any().item() == False, "Loss contains Inf values!"

        # --- Masked Mean for Motion Embeddings ---
        # Expand mask to match embedding dimensions (B, T, 1)
        motion_mask_expanded = encoder_attn_mask.unsqueeze(-1) 
        # Zero out padding frames, sum them up, and divide by actual sequence lengths
        motion_embeddings = (motion_embeds * motion_mask_expanded).sum(dim=1) / motion_mask_expanded.sum(dim=1).clamp(min=1e-9)

        # --- Masked Mean for Text Embeddings ---
        # Create a mask for the target text (where target is not padding)
        tgt_mask = (tgt != 0).float().unsqueeze(-1)
        text_embeds_raw = self.lm.shared(tgt)
        text_embeddings = (text_embeds_raw * tgt_mask).sum(dim=1) / tgt_mask.sum(dim=1).clamp(min=1e-9)
                
        return {
            "outputs": outputs,
            "motion_embeddings": motion_embeddings,
            "text_embeddings": text_embeddings,
            "loss": outputs.loss,
            "logits": outputs.logits
        }