import numpy as np
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int = 256, num_heads: int = 4):
        """
        
        """
        super(MultiHeadAttention, self).__init__()

        assert model_dim % num_heads == 0, "Model's dimension must be divisible by num_heads"

        self.model_dim = model_dim  # dimension of the model
        self.num_heads = num_heads  # Number of attention heads
        self.head_dim = model_dim // num_heads  # Dimension of each Query, Key and Value vectors

        self.query = nn.Linear(model_dim, model_dim)  # Query vector
        self.key = nn.Linear(model_dim, model_dim)  # Key vector
        self.value = nn.Linear(model_dim, model_dim)  # Value vector
        self.output = nn.Linear(model_dim, model_dim)  # Output vector

    def scaled_dot_prod(self, Q, K, V, mask=None):
        # Compute attention scores
        atten_scores = torch.matmul(Q, K.transpose(-2, -1))

        # Scale and Softmax applied to scores
        n = np.sqrt(self.head_dim)
        atten_scores = atten_scores / n

        if mask is not None:
            _MASKING_VALUE = -1e+15 if atten_scores.dtype == torch.float32 else -1e+4
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, seq_len)
            atten_scores = atten_scores.masked_fill(mask == 0, _MASKING_VALUE)

        # Attention probabilities
        atten_prob = nn.functional.softmax(atten_scores, dim=-1)

        # Matmul between attention weights and Value
        atten_w = torch.matmul(atten_prob, V)
        return atten_w
    
    def split_heads(self, x):
        # Reshape the input into (batch_size, num_heads, seq_len, head_dim)
        # enables the model to process multiple attention heads and parrallel
        # computation
        batch_size, seq_len, model_dim = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        # After applying attention to each head, the results are combined into
        # a single tensor of shape (batch_size, seq_len, model_dim)
        batch_size, _, seq_len, head_dim = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dim)

    def forward(self, Q, K, V, mask=None):
        # Apply linear layer
        Q = self.query(Q)
        K = self.key(K)
        V = self.value(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Compute scaled attention
        atten_output = self.scaled_dot_prod(Q, K, V, mask)
        atten_output = self.combine_heads(atten_output)

        # Combine backs heads and apply final linear layer
        output = self.output(atten_output)
        return output


class PositionalEmbedding(nn.Module):
    def __init__(self, model_dim: int):
        """
        Compute dynamicalely the positional encoding for input sequences.

        Params
        -------
        model_dim : int
            model's input dimension
        """
        super(PositionalEmbedding, self).__init__()
        self.model_dim = model_dim
            
    def _compute_pe(self, x):
        """
        Compute the positional encoding for input tensor x.

        Params
        -------
        x: torch.Tensor
            input tensor of shape (batch_size, seq_len, model_dim)
        """
        seq_len = x.size(1)
        device = x.device

        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)  # Tensor containing the position indices for each position in the sequences
        div_term = torch.exp(torch.arange(0, self.model_dim, 2, device=device).float() * -(math.log(10000.0) / self.model_dim))
        
        # Scaling the position indices
        pe = torch.zeros(seq_len, self.model_dim, device=device)  # Tensor later filled with positional encodings
        pe[:, 0::2] = torch.sin(position * div_term)  # sin applied to the even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # cos applied to the odd indices
        pe = pe.unsqueeze(0)
    
        return pe
        
    def forward(self, x):
        x = x + self._compute_pe(x)
        return x
    

class FeedForward(nn.Module):
    def __init__(self, model_dim: int, ff_dim: int):
        super(FeedForward, self).__init__()

        self.linear_1 = nn.Linear(model_dim, ff_dim)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(ff_dim, model_dim)


    def forward(self, x):
        feed_fwrd = self.linear_1(x)
        feed_fwrd = self.relu(feed_fwrd)
        feed_fwrd = self.linear_2(feed_fwrd)
        return feed_fwrd