
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttentionBlock(nn.Module):
    def __init__(self, key_dim: int, query_dim: int, mlp_dim: int, num_heads: int = 8):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        
        self.norm1 = nn.LayerNorm(query_dim)
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim)
        self.v_proj = nn.Linear(key_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        self.scale = self.head_dim ** -0.5
        
        self.norm2 = nn.LayerNorm(query_dim)
        self.mlp = nn.Sequential(
            nn.Linear(query_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, query_dim),
        )

    def forward(self, keys: torch.Tensor, q: torch.Tensor, mask: torch.Tensor = None, return_scores: bool = False):
        # q shape: (B, T, query_dim), keys shape: (B, S, key_dim)
        B, T, _ = q.shape

        q_normalized = self.norm1(q)
        q_proj = self.q_proj(q_normalized)  # (B, T, query_dim)
        k = self.k_proj(keys)               # (B, S, query_dim)
        v = self.v_proj(keys)               # (B, S, query_dim)

        # Reshape for multi-head: split query_dim into (num_heads, head_dim)
        q_proj = q_proj.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)             # (B, num_heads, S, head_dim)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)             # (B, num_heads, S, head_dim)

        # Compute scaled dot-product attention
        scores = torch.matmul(q_proj, k.transpose(-2, -1)) * self.scale             # (B, num_heads, T, S)
        if mask is not None:
            # Mask shape adjustment: (B, S) -> (B, 1, 1, S)
            scores = scores.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float("-1e9"))
        attn = F.softmax(scores, dim=-1)                                             # (B, num_heads, T, S)
        out = torch.matmul(attn, v)                                                  # (B, num_heads, T, head_dim)

        # Combine multiple heads
        out = out.transpose(1, 2).reshape(B, T, -1)                                  # (B, T, query_dim)
        out = self.out_proj(out)
        
        x = q + out
        x2 = self.norm2(x)
        out_final = x + self.mlp(x2)
        if return_scores:
            return out_final, scores
        return out_final


class PointAttention(nn.Module):
    def __init__(self, key_dim: int, query_dim: int, mlp_dim: int, num_layers: int, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(key_dim, query_dim, mlp_dim, num_heads) for _ in range(num_layers)]
        )
        self.query_dim = query_dim


    def forward(self, keys: torch.Tensor, mask: torch.Tensor, q_init: torch.Tensor, return_scores: bool = False):
        x = q_init
        scores_list = [] if return_scores else None
        for layer in self.layers:
            if return_scores:
                x, scores = layer(keys, x, mask, return_scores=True)
                scores_list.append(scores)
            else:
                x = layer(keys, x, mask, return_scores=False)
        x = F.layer_norm(x, x.shape[-1:])
        if return_scores:
            scores_list = torch.stack(scores_list, dim=1)
            return x, scores_list
        return x
    
class TransformerEncoder(nn.Module):
    """
    A transformer encoder that stacks multiple self-attention and MLP layers.
    Each layer is implemented using CrossAttentionBlock for self-attention.
    """
    def __init__(self, dim: int, mlp_dim: int, num_layers: int,num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(dim, dim, mlp_dim,num_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, dim)
            mask: Optional boolean mask of shape (batch, seq_len), where True indicates valid tokens.
        Returns:
            Tensor of same shape as x.
        """
        for layer in self.layers:
            x = layer(x, x, mask)
        return x
    
class TransformerMLP(nn.Module):
    """
    A transformer encoder-like module that stacks multiple MLP layers without attention.
    Each layer consists of a LayerNorm and an MLP residual block.
    """
    def __init__(self, dim: int, mlp_dim: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, dim),
            )
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, pad_mask=None) -> torch.Tensor:
        for mlp in self.layers:
            x = x + mlp(x)
        return x
