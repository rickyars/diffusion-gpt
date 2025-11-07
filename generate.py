"""
Inference script for generating text from trained discrete diffusion GPT models.
"""

import argparse
import os
import pickle
import sys
import textwrap

import torch
import yaml

from model import GPT, GPTConfig
from utils import (
    GeometricNoise,
    decode,
    sample_categorical,
    set_seed,
    staggered_score,
    transition,
)


def print_wrapped(text: str, width: int = 80):
    """Print text wrapped to a maximum line width."""
    paragraphs = text.splitlines()
    wrapped = [textwrap.fill(p, width=width) if p else '' for p in paragraphs]
    final_text = "\n".join(wrapped)
    print(final_text)


def generate_samples(
    model: GPT,
    noise: GeometricNoise,
    vocab_size: int,
    itos: dict,
    num_samples: int = 10,
    context_length: int = 256,
    steps: int = 128,
    device: torch.device = torch.device('cpu'),
    eps: float = 1e-5,
    verbose: bool = False,
):
    """
    Generate text samples from a trained discrete diffusion model.

    Args:
        model: Trained GPT model
        noise: GeometricNoise instance
        vocab_size: Vocabulary size
        itos: Index to string mapping
        num_samples: Number of samples to generate
        context_length: Length of generated sequences
        steps: Number of denoising steps
        device: torch device
        eps: Small epsilon for numerical stability
        verbose: Print intermediate denoising steps

    Returns:
        List of generated text samples
    """
    model.eval()
    samples = []

    step_size = (1 - eps) / steps
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    with torch.no_grad():
        for sample_idx in range(num_samples):
            print(f"\nGenerating sample {sample_idx + 1}/{num_samples}...")

            # Start with random tokens
            x = torch.randint(0, vocab_size, (1, context_length), device=device)

            if verbose:
                print("\nInitial random text:")
                print_wrapped(decode(x[0], itos))
                print("\nDenoising...")

            # Denoising loop
            for i in range(steps + 1):
                t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
                curr_sigma_bar = noise(t)[0]

                if i < steps:
                    next_sigma_bar = noise(t - step_size)[0]
                    delta_sigma = curr_sigma_bar - next_sigma_bar

                    log_score = model(x, curr_sigma_bar)
                    score = torch.exp(log_score)

                    stag_score = staggered_score(score, delta_sigma)
                    probs = stag_score * transition(x, delta_sigma, vocab_size)
                    x = sample_categorical(probs)
                else:
                    # Last denoising step
                    delta_sigma = curr_sigma_bar

                    log_score = model(x, curr_sigma_bar)
                    score = torch.exp(log_score)

                    stag_score = staggered_score(score, delta_sigma)
                    probs = stag_score * transition(x, delta_sigma, vocab_size)
                    x = sample_categorical(probs)

                if verbose and (i % (steps // 5) == 0 or i == steps):
                    print(f"\nStep {i}/{steps}:")
                    print_wrapped(decode(x[0], itos))

            # Decode final sample
            generated_text = decode(x[0], itos)
            samples.append(generated_text)

            if not verbose:
                print("\nGenerated text:")
                print_wrapped(generated_text)

    return samples


def main():
    parser = argparse.ArgumentParser(description='Generate text from trained discrete diffusion GPT models')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--vocab', type=str, default=None,
                       help='Path to vocabulary pickle file. If not specified, infers from model name.')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of samples to generate. Overrides config.')
    parser.add_argument('--steps', type=int, default=None,
                       help='Number of denoising steps. Overrides config.')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file to save generated samples. If not specified, prints to stdout.')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed. Overrides config.')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Overrides config.')
    parser.add_argument('--verbose', action='store_true',
                       help='Print intermediate denoising steps')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed
    seed = args.seed if args.seed is not None else config['seed']
    set_seed(seed)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])

    print(f"Using device: {device}")

    # Load model checkpoint
    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found: {args.model}")
        sys.exit(1)

    print(f"Loading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    # Extract model config
    if 'config' in checkpoint:
        model_config_dict = checkpoint['config']
        model_config = GPTConfig(**model_config_dict)
    else:
        # Fallback: use config from yaml
        print("Warning: Model config not found in checkpoint, using config.yaml")
        model_config = GPTConfig(
            block_size=config['model']['context_length'],
            vocab_size=checkpoint.get('vocab_size', 65),
            n_layer=config['model']['n_layer'],
            n_head=config['model']['n_head'],
            n_embd=config['model']['n_embd'],
            cond_dim=config['model']['cond_dim'],
            dropout=0.0,  # no dropout during inference
            bias=config['model']['bias'],
        )

    vocab_size = checkpoint.get('vocab_size', model_config.vocab_size)

    # Initialize model
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")

    # Load vocabulary
    if args.vocab:
        vocab_path = args.vocab
    else:
        # Infer vocab path from model name
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        # Remove _epoch_N suffix if present
        if '_epoch_' in model_name:
            model_name = model_name.split('_epoch_')[0]
        vocab_path = os.path.join(config['paths']['vocab_dir'], f"{model_name}_vocab.pkl")

    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found: {vocab_path}")
        print("Please specify vocabulary file with --vocab")
        sys.exit(1)

    print(f"Loading vocabulary from: {vocab_path}")
    with open(vocab_path, 'rb') as f:
        vocab_meta = pickle.load(f)
        itos = vocab_meta['itos']
        stoi = vocab_meta['stoi']

    # Initialize noise schedule
    noise = GeometricNoise(
        sigma_min=config['noise']['sigma_min'],
        sigma_max=config['noise']['sigma_max'],
    )

    # Generation parameters
    num_samples = args.samples if args.samples is not None else config['sampling']['num_samples']
    steps = args.steps if args.steps is not None else config['sampling']['steps']
    context_length = model_config.block_size

    print(f"\nGenerating {num_samples} samples with {steps} denoising steps...")
    print(f"Context length: {context_length} characters\n")

    # Generate samples
    samples = generate_samples(
        model=model,
        noise=noise,
        vocab_size=vocab_size,
        itos=itos,
        num_samples=num_samples,
        context_length=context_length,
        steps=steps,
        device=device,
        verbose=args.verbose,
    )

    # Save to file if specified
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            for i, sample in enumerate(samples):
                f.write(f"=== Sample {i+1} ===\n")
                f.write(sample)
                f.write("\n\n")
        print(f"\nSaved {num_samples} samples to: {args.output}")
    else:
        print(f"\n{'='*80}")
        print("Generation complete!")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
