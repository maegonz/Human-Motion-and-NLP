import torch
import torch.nn as nn
from .utils import linear_beta_schedule
from .stblock import StTBlock
from ..motion2text.transformers.blocks import FeedForward, PositionalEmbedding
from tqdm  import tqdm
from transformers import AutoTokenizer

class StTDiffusion(nn.Module):
    """
    Spatio-temporal Transformer Graph Diffusion model with text conditioning.

    This model takes in a noisy motion sequence and a text embedding,
    then predicts the noise to be removed, in order to denoise the motion
    sequence. The model uses a transformer-based architecture for denoising, and incorporates
    text embeddings as additional channels to condition the denoising process.
    """

    def __init__(self, 
                 timesteps: int=1000, 
                 vocab_size: int=10000, 
                 input_dim: int=3,
                 model_dim: int=512,
                 num_heads: int=8,
                 swiglu: bool=False, 
                 dropout: float=0.1,
                 tokenizer_name: str="bert-base-uncased"):
        super(StTDiffusion, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.positional_embedding = PositionalEmbedding(model_dim)

        # Motion embedding
        self.motion_embedding = FeedForward(input_dim=input_dim, model_dim=model_dim, ff_dim=model_dim, dropout=dropout)

        # Caption embedding
        self.caption_embedding = nn.Sequential(
            nn.Embedding(vocab_size, model_dim),
            FeedForward(input_dim=model_dim, model_dim=model_dim, ff_dim=model_dim, dropout=dropout)
        )

        # Noise schedule
        self.betas = linear_beta_schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # Spatio-temporal transformer for denoising
        self.sttd =nn.ModuleList([
            StTBlock(model_dim, model_dim, num_heads, dropout, swiglu)  # Add text channels
        ])

        # Timestep embedding
        self.timesteps = timesteps
        self.timestep_embedding = FeedForward(input_dim=1, model_dim=model_dim, ff_dim=model_dim, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, 
                m: torch.Tensor, 
                t: torch.Tensor, 
                token_ids: torch.Tensor):
        """
        Forward pass of the diffusion model

        Params
        -------
        m: torch.Tensor[batch_size, seq_len, n_joints, spatial_dim]
            Noisy motion sequences
        t: torch.Tensor[batch_size]
            Timesteps 
        token_ids: torch.Tensor[batch_size, seq_len]
            Token IDs for the text captions
        """

        _, seq_len, n_joints, spatial_dim = m.shape

        # Embed the motion data
        m = self.motion_embedding(m)  # [batch_size, seq_len, n_joints, motion_embed_dim]
        m = self.positional_embedding(m)
        m = self.dropout(m)

        # Get caption embeddings
        caption = self.caption_embedding(token_ids)  # [batch_size, seq_len, text_embed_dim]

        # Add timestep information
        t = self.timestep_embedding(t.unsqueeze(1).float())  # [batch_size, text_embed_dim]

        # Combine timestep and caption information
        c = t + caption  # [batch_size, text_embed_dim]

        # Reshape conditioning to spatial dimensions and concatenate with image
        c = c.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [batch_size, text_embed_dim, 1, 1]
        c = c.repeat(1, 1, seq_len, n_joints, spatial_dim)  # [batch_size, text_embed_dim, seq_len, n_joints, spatial_dim]

        # Predict noise
        for st_block in self.sttd:
            m = st_block(m, c)  # [batch_size, seq_len, n_joints, spatial_dim]

        return m

    def sample_timesteps(self, batch_size):
        """Sample random timesteps for training"""
        return torch.randint(0, self.timesteps, (batch_size,))

    @torch.no_grad()
    def reverse(self, captions, seq_len, num_motions=4, device="cuda"):
        """
        Generate motion sequences from text captions using the trained diffusion model
        """
        self.eval()

        # Convert captions to embeddings
        tokens = self.tokenizer(captions, max_length=512, truncation=True, return_tensors="pt")
        caption_ids = tokens['input_ids'].to(device)

        # Start from random noise
        x = torch.randn(num_motions, seq_len, 22, 3, device="cuda")

        # Reverse diffusion process
        for t in tqdm(reversed(range(self.timesteps)), desc="Generating images"):
            t_batch = torch.full((num_motions,), t, device=device, dtype=torch.long)

            # Predict noise
            predicted_noise = self(x, t_batch, caption_ids)

            # Get alpha and beta parameters
            alpha_t = self.alphas[t]
            alpha_t_cumprod = self.alphas_cumprod[t]
            beta_t = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # Reverse process step
            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_t_cumprod)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise

        # Denormalize images
        motions = x

        return motions, captions