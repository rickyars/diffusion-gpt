"""
Discrete Diffusion GPT Model Architecture.
Adapted from nanoGPT and Score-Entropy-Discrete-Diffusion.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Configuration for the Discrete Diffusion GPT model."""
    block_size: int = 256  # maximum sequence length
    vocab_size: int = 65  # vocabulary size
    n_layer: int = 6  # number of transformer layers
    n_head: int = 6  # number of attention heads
    n_embd: int = 384  # embedding dimension
    cond_dim: int = 64  # conditioning dimension for noise
    dropout: float = 0.2  # dropout probability
    bias: bool = False  # use bias in linear layers


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulate input with shift and scale."""
    return x * (1 + scale) + shift


def bias_add_scale(
    x: torch.Tensor,
    bias: Optional[torch.Tensor],
    scale: torch.Tensor,
    residual: Optional[torch.Tensor]
) -> torch.Tensor:
    """Apply bias, scale, and residual connection."""
    if bias is not None:
        out = scale * (x + bias)
    else:
        out = scale * x

    if residual is not None:
        out = residual + out
    return out


class MLP(nn.Module):
    """Multi-layer perceptron with GELU activation."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    """
    Multi-head self-attention without causal masking.
    Unlike autoregressive GPT, this can attend to all positions.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention support
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality

        # calculate query, key, values for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attention (non-causal)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class DDiTBlock(nn.Module):
    """
    Discrete Diffusion Transformer Block.
    Combines self-attention and MLP with adaptive layer normalization (adaLN).
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

        # Adaptive layer norm modulation from noise conditioning
        self.adaLN_modulation = nn.Linear(config.cond_dim, 6 * config.n_embd)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_embd) input tensor
            c: (B, cond_dim) conditioning (noise level)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c)[:, None].chunk(6, dim=2)

        x_skip = x
        x = modulate(self.ln_1(x), shift_msa, scale_msa)
        x = self.attn(x)
        x = bias_add_scale(self.attn(self.ln_1(x)), None, gate_msa, x_skip)
        x = bias_add_scale(self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp)), None, gate_mlp, x)
        return x


class DDitFinalLayer(nn.Module):
    """Final layer mapping to vocabulary logits with adaptive normalization."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.norm_final = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.linear = nn.Linear(config.n_embd, config.vocab_size)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(config.cond_dim, 2 * config.n_embd)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations using sinusoidal embeddings.
    """

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: 1-D Tensor of timesteps
            dim: embedding dimension
            max_period: controls the minimum frequency

        Returns:
            (N, D) Tensor of positional embeddings
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class GPT(nn.Module):
    """
    Discrete Diffusion GPT Model.
    Character-level transformer that takes noisy sequences and noise levels as input,
    outputs log probability ratios for denoising.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.sigma_map = TimestepEmbedder(config.cond_dim)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([DDiTBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = DDitFinalLayer(config)

        # initialize weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Return the number of parameters in the model.
        For non-embedding count (default), position embeddings get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            idx: (B, T) tensor of input token indices
            sigma: (B,) or (B, 1) tensor of noise levels

        Returns:
            (B, T, vocab_size) tensor of log probability ratios
        """
        sigma = sigma.reshape(-1)
        device = idx.device
        b, t = idx.size()
        c = F.silu(self.sigma_map(sigma))
        assert t <= self.config.block_size, \
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x, c)
        x = self.transformer.ln_f(x)

        # compute log probability ratios
        x = self.lm_head(x, c)
        # zero out the score for the current token (diagonal in rate matrix)
        x = torch.scatter(x, -1, idx[..., None], torch.zeros_like(x[..., :1]))

        return x
