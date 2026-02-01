import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from .blocks import PositionalEmbedding
from .encoders import Encoder

class TransfoLM(nn.Module):
    def __init__(self,
                 motion_dim: int,
                 lm_name: str = 't5-small',
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
        self.enc_embedding = nn.Linear(motion_dim, model_dim)
        self.pos_embedding = PositionalEmbedding(model_dim)
        
        # Encoder layers
        self.encoder = nn.ModuleList(
            [Encoder(model_dim, num_heads, dropout, ff_dim) for _ in range(num_layers)]
        )

        # Tokenizer for sentence generation
        self.tokenizer = AutoTokenizer.from_pretrained(lm_name, use_fast=True)

        # Decoder layers
        self.lm = T5ForConditionalGeneration.from_pretrained(lm_name)
        # Freeze LM parameters
        for param in self.lm.parameters():
            param.requires_grad = False
        # self.lm.eval()
        self.lm.config.use_cache = False  # Disable caching for training

        # Projection layer to map encoder output to LM model dimension
        self.projection = nn.Linear(model_dim, self.lm.config.d_model)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, src, tgt, encoder_attn_mask=None, t5_attn_mask=None, generation=False):
        """
        Params
        -------
        src: torch.Tensor
            input to the encoder, which is the motion
        tgt: torch.Tensor
            input to the decoder, which is the text
        encoder_attn_mask: torch.Tensor, optional
            attention mask for the encoder, by default None
        t5_attn_mask: torch.Tensor, optional
            attention mask for the T5 decoder, by default None
        generation: bool, optional
            whether generate sentences or not, by default False

        Returns
        -------
        returns output logits from the model
        """
        # Embedding and positional encoding
        # src: (B, T, 22, 3)
        B, T, _, _ = src.shape

        src = src.view(B, T, 66)           # flatten joints
        src_emb = self.enc_embedding(src)  # (B, T, model_dim)
        src_emb = self.pos_embedding(src_emb)
        src_emb = self.dropout(src_emb)

        # Encoder
        encoder_output = src_emb
        for encoder in self.encoder:
            encoder_output = encoder(encoder_output, mask=encoder_attn_mask)

        # Project encoder output to LM model dimension
        encoder_output = self.projection(encoder_output)
        # Wrap for T5 compatibility
        encoder_output = BaseModelOutput(last_hidden_state=encoder_output)
        
        # Teacher forcing during training
        tgt = tgt.long()
        tgt_input_ids = tgt[:, :-1].contiguous()
        tgt_labels = tgt[:, 1:].contiguous()
        if t5_attn_mask is not None:
            decoder_attn_mask = t5_attn_mask[:, :-1].contiguous()

        if generation:
            # Generation mode
            outputs_ids = self.lm.generate(
                encoder_outputs=encoder_output,
                max_length=50,
                num_beams=4,
                early_stopping=True,
                output_scores=True,
                no_repeat_ngram_size=2,   # Prevents any 2-word phrase from appearing twice
                repetition_penalty=2.0,   # Penalizes words that have already been generated
                length_penalty=1.0)        # encourages the model to generate meaningful length
                        
            outputs = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True) 
            return outputs
        
        else:
            # Decoder forward pass using T5 LM
            outputs = self.lm(encoder_outputs=encoder_output,
                            attention_mask=encoder_attn_mask,
                            decoder_input_ids=tgt_input_ids,
                            decoder_attention_mask=decoder_attn_mask,
                            labels=tgt_labels,
                            return_dict=True)
        
        return outputs  # outputs.logits, outputs.loss