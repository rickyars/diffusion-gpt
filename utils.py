"""
Utility functions for discrete diffusion text generation.
Includes noise schedules, perturbation functions, and text decoding.
"""

import torch
import numpy as np
import random


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def encode(text: str, stoi: dict) -> list:
    """
    Encodes a string to a list of token indices.

    Args:
        text: Input text string
        stoi: Dictionary mapping characters to indices

    Returns:
        List of character indices
    """
    return [stoi[ch] for ch in text if ch in stoi]


def decode(indices_tensor: torch.Tensor, itos: dict):
    """
    Decodes a 1D tensor of indices to text.

    Args:
        indices_tensor: Tensor of character indices
        itos: Dictionary mapping indices to characters

    Returns:
        Decoded string
    """
    indices = indices_tensor.cpu().numpy()
    return ''.join([itos[int(i)] for i in indices])


def perturb_batch(batch: torch.Tensor, sigma_bar: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Diffuse each token independently according to the discrete diffusion process.

    With probability e^{-sigma_bar} + (1 - e^{-sigma_bar})/N, a token stays the same.
    Otherwise, it jumps uniformly to one of the other N-1 tokens.

    Args:
        batch: LongTensor of shape [B, L], each entry in [0, vocab_size-1]
        sigma_bar: scalar tensor or tensor of shape [B, 1]
        vocab_size: number of tokens in vocabulary

    Returns:
        batch_pert: perturbed batch of LongTensor
    """
    B, L = batch.shape

    # Compute move probability: (1 - e^{-sigma}) * (1 - 1/N)
    stay_base = torch.exp(-sigma_bar)
    move_prob = (1 - stay_base) * (1 - 1 / vocab_size)

    # Bernoulli: should this token move?
    move_mask = torch.rand(B, L, device=batch.device) < move_prob

    # For tokens that move, sample a different id uniformly from the other N-1 ids
    # Sample r in [0, N-2], then map to [0..N-1]\{orig} by skipping the original
    r = torch.randint(low=0, high=vocab_size - 1, size=(B, L), device=batch.device)
    # shift up by 1 wherever r >= original id, covering {0, .., k-1, k+1, .., N-1}
    new_ids = r + (r >= batch)

    # Apply moves; else keep original
    batch_pert = torch.where(move_mask, new_ids, batch)
    return batch_pert


class GeometricNoise:
    """
    Geometric noise schedule for discrete diffusion.

    Defines sigma(t) and integrated noise bar_sigma(t) = integral_0^t sigma(tau) dtau
    """
    def __init__(self, sigma_min: float = 1e-4, sigma_max: float = 20.0):
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])

    def rate_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Returns sigma(t)"""
        return (self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t *
                (self.sigmas[1].log() - self.sigmas[0].log()))

    def total_noise(self, t: torch.Tensor) -> torch.Tensor:
        """Returns bar_sigma(t)"""
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t

    def __call__(self, t: torch.Tensor):
        """
        Returns bar_sigma(t) and sigma(t)

        Args:
            t: time step(s), tensor of shape [B] or scalar

        Returns:
            (bar_sigma, sigma) tuple
        """
        return self.total_noise(t), self.rate_noise(t)


def transition(x_t: torch.Tensor, delta_sigma: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Forward transition kernel: exp(sigma_t^Delta Q^{tok})(x_t, y)

    Approximates the finite-time forward diffusion probability of moving from token x_t to y
    after a noise increment of Delta_sigma.

    Args:
        x_t: (B, L) integer tensor of current tokens
        delta_sigma: scalar tensor representing sigma_t^{Delta}
        vocab_size: vocabulary size

    Returns:
        trans_probs: (B, L, V) tensor of categorical probabilities over next tokens
    """
    # Uniform mixing term from exp(delta_sigma * Q^{tok})
    base_prob = (1 - torch.exp(-delta_sigma[..., None])) / vocab_size
    trans = torch.ones(*x_t.shape, vocab_size, device=x_t.device) * base_prob

    # Remove the uniform contribution for the current token
    trans = trans.scatter(-1, x_t[..., None], torch.zeros_like(trans))

    # Ensure that probabilities across the vocabulary sum to 1
    diag_fill = 1 - trans.sum(dim=-1, keepdim=True)
    trans = trans.scatter(-1, x_t[..., None], diag_fill)
    return trans


def staggered_score(score: torch.Tensor, delta_sigma: torch.Tensor) -> torch.Tensor:
    """
    Applies the inverse exponential operator: exp(-sigma_t^Delta Q^{tok}) s_theta(x_t, t)

    This "staggered" score correction accounts for the finite time-step Delta_t.

    Args:
        score: (B, L, V) tensor, model output s_theta(x_t, t)
        delta_sigma: scalar tensor representing sigma_t^{Delta}

    Returns:
        adjusted_score: (B, L, V) tensor, transformed score
    """
    vocab_size = score.shape[-1]
    exp_factor = torch.exp(-delta_sigma)[..., None]  # (B, L, 1) or scalar
    correction = ((exp_factor - 1) / (vocab_size * exp_factor)) * score.sum(dim=-1, keepdim=True)
    return correction + score / exp_factor


def compute_loss_weight(sigma: torch.Tensor, mode: str = 'uniform') -> torch.Tensor:
    """
    Compute loss weights based on noise level (sigma) and weighting mode.

    This implements loss weighting strategies from D3PM and EDM papers to improve
    sample quality (diversity, sharpness) without performance overhead.

    Args:
        sigma: (B,) tensor of noise levels at current timestep
        mode: weighting mode, one of:
            - 'uniform': No weighting, returns all 1s (baseline behavior)
            - 'snr': Signal-to-Noise Ratio weighting, emphasizes low noise (clean data)
                     Formula: weight = 1.0 / (sigma^2 + 1.0)
                     Effect: Improves sharpness and fine details
                     Reference: EDM (Karras et al. 2022)
            - 'importance': Importance sampling, emphasizes middle timesteps
                            Formula: weight = sigma / (sigma + 1.0)
                            Effect: Promotes faster convergence, stronger learning signal
                            Reference: D3PM (Austin et al. 2021)

    Returns:
        weight: (B,) tensor of loss weights, same shape as input sigma

    Raises:
        ValueError: if mode is not one of 'uniform', 'snr', 'importance'

    References:
        - D3PM: Discrete Diffusion Probabilistic Models (https://arxiv.org/abs/2107.03006)
        - EDM: Elucidating the Design Space of Diffusion-Based Generative Models (https://arxiv.org/abs/2206.00364)
    """
    if mode == 'uniform':
        return torch.ones_like(sigma)
    elif mode == 'snr':
        # SNR weighting: emphasizes denoising at low noise levels
        # Formula: weight = 1 / (sigma^2 + 1)
        return 1.0 / (sigma ** 2 + 1.0)
    elif mode == 'importance':
        # Importance sampling weighting: emphasizes middle timesteps
        # Formula: weight = sigma / (sigma + 1)
        return sigma / (sigma + 1.0)
    else:
        raise ValueError(
            f"Unknown loss weighting mode: '{mode}'. "
            f"Must be one of: 'uniform', 'snr', 'importance'"
        )


def sample_categorical(probs: torch.Tensor) -> torch.Tensor:
    """
    Sample from a batch of categorical distributions using the Gumbel-max trick.

    Args:
        probs: (B, L, V) tensor of probabilities that sum to 1 along dim=-1

    Returns:
        samples: (B, L) tensor of sampled token indices
    """
    # Add a small epsilon for numerical stability
    eps = 1e-10
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(probs) + eps) + eps)
    return torch.argmax(torch.log(probs + eps) + gumbel_noise, dim=-1)
