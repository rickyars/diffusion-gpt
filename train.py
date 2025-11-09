"""
Training script for discrete diffusion GPT models.
Supports training on single dataset or multiple datasets sequentially.
"""

import argparse
import os
import signal
import sys
import time
from typing import Optional

import torch
import torch.optim as optim
import yaml

from dataset_loader import get_data_loader
from model import GPT, GPTConfig
from utils import GeometricNoise, perturb_batch, set_seed


def score_entropy(
    score_log: torch.Tensor,
    sigma_bar: torch.Tensor,
    x_t: torch.Tensor,
    x0: torch.Tensor,
    vocab_size: int,
    clamp_exp: float = 30.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Compute the Score Entropy Loss (DWDSE) without outer sigma_t multiplier.

    Args:
        score_log: (B, L, V) tensor of model outputs = log s_theta(x_t, bar{sigma}_t)
        sigma_bar: (B, 1) tensor for bar_sigma_t (integrated noise)
        x_t: (B, L) int tensor with current noised tokens
        x0: (B, L) int tensor with original clean tokens
        vocab_size: vocabulary size
        clamp_exp: clamp for exponent to keep exp(score_log) stable
        eps: small constant for numerical stability

    Returns:
        loss: (B, L) tensor containing loss per token position
    """
    B, L, V = score_log.shape

    # Precompute helpers
    esigm1 = torch.where(
        sigma_bar < 0.5,
        torch.expm1(sigma_bar),
        torch.exp(sigma_bar) - 1
    )

    ratio = esigm1 / (esigm1 + vocab_size)
    ratio = torch.clamp(ratio, min=eps)

    # Clamp and exponentiate score
    score_log = torch.clamp(score_log, max=clamp_exp)
    s = torch.exp(score_log)

    def take_at(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return torch.gather(logits, dim=-1, index=idx[..., None]).squeeze(-1)

    # Build positive term
    s_scaled = s / (vocab_size - 1)
    s_mean_all = s_scaled.sum(dim=-1)
    s_at_xt = take_at(s_scaled, x_t)
    pos_term = s_mean_all - s_at_xt

    # Build negative term
    log_s_mean = score_log.sum(dim=-1) / (vocab_size - 1)
    log_s_at_xt = take_at(score_log, x_t) / (vocab_size - 1)
    base_neg = log_s_mean - log_s_at_xt

    no_move = (x_t == x0)
    neg_term_no_move = ratio * base_neg
    neg_term_move = take_at(score_log, x0) / (ratio * (vocab_size - 1)) + \
                    (vocab_size - 2) * base_neg / (vocab_size - 1)
    neg_term = torch.where(no_move, neg_term_no_move, neg_term_move)

    # Build constant term
    const_no_move = ratio * (torch.log(ratio) - 1.0)
    const_move = ((-torch.log(ratio) - 1.0) / ratio - (vocab_size - 2)) / (vocab_size - 1)
    const_term = torch.where(no_move, const_no_move, const_move)

    loss = pos_term - neg_term + const_term
    return loss


def loss_function(
    model: GPT,
    x0: torch.Tensor,
    noise: GeometricNoise,
    vocab_size: int,
    t: Optional[torch.Tensor] = None,
    x_t: Optional[torch.Tensor] = None,
    sampling_eps: float = 1e-3,
) -> torch.Tensor:
    """
    Computes the loss for a batch of data.

    Args:
        model: discrete diffusion model
        x0: (B, L) LongTensor of original clean tokens
        noise: GeometricNoise instance
        vocab_size: vocabulary size
        t: (B,) float tensor with time steps in [0, 1]. If None, sampled uniformly.
        x_t: (B, L) int tensor with perturbed tokens. If None, generated on-the-fly.
        sampling_eps: small epsilon to avoid 0 or 1 time steps

    Returns:
        loss: scalar tensor with the loss
    """
    if t is None:
        t = (1 - sampling_eps) * torch.rand(x0.shape[0], device=x0.device) + sampling_eps

    sigma_bar, sigma = noise(t)

    if x_t is None:
        x_t = perturb_batch(x0, sigma_bar[:, None], vocab_size)

    log_score = model(x_t, sigma_bar)
    loss = score_entropy(log_score, sigma_bar[:, None], x_t, x0, vocab_size)
    loss = (sigma[:, None] * loss).mean(dim=-1).mean()

    return loss


def train_model(
    dataset_name: str,
    dataset_path: str,
    config: dict,
    device: torch.device,
    resume_from: Optional[str] = None,
    dataset_config: Optional[dict] = None,
):
    """
    Train a discrete diffusion model on a single dataset.

    Args:
        dataset_name: Name of the dataset (for saving checkpoints)
        dataset_path: Path to .txt file
        config: Configuration dictionary
        device: torch device
        resume_from: Optional path to checkpoint to resume from
        dataset_config: Optional dataset-specific configuration (for max_chars, etc.)
    """
    print(f"\n{'='*80}")
    print(f"Training model on: {dataset_name}")
    print(f"Dataset path: {dataset_path}")
    print(f"{'='*80}\n")

    # Load configuration
    batch_size = config['training']['batch_size']
    learning_rate = config['training']['learning_rate']
    epochs = config['training']['epochs']
    context_length = config['model']['context_length']
    val_split = config['training']['val_split']
    eval_interval = config['training']['eval_interval']
    save_interval = config['training']['save_interval']
    log_interval = config['training']['log_interval']
    use_compile = config['training'].get('use_compile', False)  # Default to False for compatibility

    # Paths
    models_dir = config['paths']['models_dir']
    vocab_dir = config['paths']['vocab_dir']
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(vocab_dir, exist_ok=True)

    vocab_path = os.path.join(vocab_dir, f"{dataset_name}_vocab.pkl")
    model_path = os.path.join(models_dir, f"{dataset_name}.pt")

    # Check if model already exists (skip completed training)
    skip_completed = config['training'].get('skip_completed', True)
    if skip_completed and os.path.exists(model_path) and not resume_from:
        print(f"✓ Model already exists: {model_path}")
        print(f"  Skipping training (set 'skip_completed: false' in config to retrain)")
        print(f"{'='*80}\n")
        return  # Skip this dataset

    # Get max_chars limit from dataset config (optional)
    max_chars = None
    if dataset_config:
        max_chars = dataset_config.get('max_chars', None)
    if max_chars:
        print(f"Dataset size limit: {max_chars:,} characters")

    # Create dataloaders
    print("Loading training data...")
    train_loader, vocab_size, itos, stoi = get_data_loader(
        data_path=dataset_path,
        batch_size=batch_size,
        context_len=context_length,
        split='train',
        val_split=val_split,
        vocab_path=vocab_path if os.path.exists(vocab_path) else None,
        shuffle=True,
        num_workers=4,  # Increased for faster data loading (adjust based on CPU cores)
        max_chars=max_chars,
    )

    print("Loading validation data...")
    val_loader, _, _, _ = get_data_loader(
        data_path=dataset_path,
        batch_size=batch_size,
        context_len=context_length,
        split='val',
        val_split=val_split,
        vocab_path=vocab_path,
        shuffle=False,
        num_workers=4,  # Increased for faster data loading
        max_chars=max_chars,
    )

    # Save vocabulary for later use
    if not os.path.exists(vocab_path):
        train_loader.dataset.save_vocab(vocab_path)

    # Initialize model
    model_config = GPTConfig(
        block_size=context_length,
        vocab_size=vocab_size,
        n_layer=config['model']['n_layer'],
        n_head=config['model']['n_head'],
        n_embd=config['model']['n_embd'],
        cond_dim=config['model']['cond_dim'],
        dropout=config['model']['dropout'],
        bias=config['model']['bias'],
    )

    model = GPT(model_config)
    model.to(device)

    print(f"\nModel parameters: {model.get_num_params() / 1e6:.2f}M")

    # Compile model for faster training (PyTorch 2.0+)
    # Note: Requires Triton, may not work on Windows
    if use_compile and hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            print("Compiling model with torch.compile() (set use_compile: false in config to disable)...")
            model = torch.compile(model)
            print("✓ Model compiled successfully - expect 30-50% faster training!")
        except Exception as e:
            print(f"⚠ torch.compile() failed: {str(e)[:100]}")
            print("  Set 'use_compile: false' in config.yaml to disable this attempt")
            print("  Continuing with eager mode (slightly slower but works fine)")
    elif not use_compile:
        print("torch.compile() disabled (use_compile: false in config)")
        print("Enable with 'use_compile: true' if you have Triton installed")

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Mixed precision training for faster computation
    use_amp = device.type == 'cuda'
    # Use torch.amp.GradScaler instead of deprecated torch.cuda.amp.GradScaler
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Using mixed precision training (AMP) for faster computation")

    # Initialize noise schedule
    noise = GeometricNoise(
        sigma_min=config['noise']['sigma_min'],
        sigma_max=config['noise']['sigma_max'],
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    # Training loop with graceful shutdown handling
    print("\nStarting training...")
    print("Press Ctrl+C to save progress and exit gracefully\n")
    print(f"Training configuration:")
    print(f"  Batches per epoch: {len(train_loader)}")
    print(f"  Samples per epoch: {len(train_loader) * batch_size}")
    print(f"  Total training steps: {len(train_loader) * epochs}")
    print()
    model.train()

    # Flag for graceful shutdown
    should_stop = False

    def signal_handler(sig, frame):
        nonlocal should_stop
        print("\n\nReceived interrupt signal. Saving checkpoint and exiting gracefully...")
        print("(Press Ctrl+C again to force quit without saving)")
        should_stop = True

    # Register signal handler for graceful shutdown
    original_handler = signal.signal(signal.SIGINT, signal_handler)

    try:
        for epoch in range(start_epoch, epochs):
            if should_stop:
                break

            epoch_start = time.time()
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(train_loader):
                if should_stop:
                    break

                batch = batch.to(device, non_blocking=True)

                # Mixed precision training
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        loss = loss_function(
                            model, batch, noise, vocab_size,
                            sampling_eps=config['noise']['sigma_min']
                        )

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = loss_function(
                        model, batch, noise, vocab_size,
                        sampling_eps=config['noise']['sigma_min']
                    )

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if (batch_idx + 1) % log_interval == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] "
                          f"Loss: {loss.item():.4f}")

            avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            epoch_time = time.time() - epoch_start
            samples_per_sec = (num_batches * batch_size) / epoch_time if epoch_time > 0 else 0
            print(f"\nEpoch [{epoch+1}/{epochs}] Average Train Loss: {avg_train_loss:.4f}")
            print(f"  Epoch time: {epoch_time:.1f}s | Throughput: {samples_per_sec:.0f} samples/sec")

            # Validation
            if (epoch + 1) % eval_interval == 0:
                model.eval()
                val_loss = 0.0
                val_batches = 0

                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        loss = loss_function(model, batch, noise, vocab_size,
                                           sampling_eps=config['noise']['sigma_min'])
                        val_loss += loss.item()
                        val_batches += 1

                avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
                print(f"Validation Loss: {avg_val_loss:.4f}\n")
                model.train()

            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': model_config.__dict__,
                    'vocab_size': vocab_size,
                    'loss': avg_train_loss,
                }
                checkpoint_path = os.path.join(models_dir, f"{dataset_name}_epoch_{epoch+1}.pt")
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}\n")

    except KeyboardInterrupt:
        # Second Ctrl+C - force quit
        print("\nForce quit - checkpoint not saved!")
        raise
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, original_handler)

    # Save final or interrupted checkpoint
    if should_stop:
        print("\nSaving interrupted training checkpoint...")
        interrupted_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model_config.__dict__,
            'vocab_size': vocab_size,
            'loss': avg_train_loss if num_batches > 0 else float('inf'),
        }
        interrupted_path = os.path.join(models_dir, f"{dataset_name}_interrupted_epoch_{epoch+1}.pt")
        torch.save(interrupted_checkpoint, interrupted_path)
        print(f"Interrupted checkpoint saved to: {interrupted_path}")
        print("You can resume training with: python train.py --dataset {dataset_name} --resume {interrupted_path}")
    else:
        # Normal completion - save final model
        final_checkpoint = {
            'epoch': epochs - 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': model_config.__dict__,
            'vocab_size': vocab_size,
            'loss': avg_train_loss,
        }
        torch.save(final_checkpoint, model_path)
        print(f"\nTraining completed! Final model saved to: {model_path}")


def main():
    parser = argparse.ArgumentParser(description='Train discrete diffusion GPT models')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset name to train on (from config). If not specified, trains on all enabled datasets.')
    parser.add_argument('--all', action='store_true',
                       help='Train on all enabled datasets sequentially')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu). Overrides config.')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed
    set_seed(config['seed'])

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])

    print(f"Using device: {device}")

    # Determine which datasets to train on
    if args.dataset:
        # Train on single dataset
        if args.dataset not in config['datasets']:
            print(f"Error: Dataset '{args.dataset}' not found in config")
            sys.exit(1)

        dataset_config = config['datasets'][args.dataset]
        dataset_path = dataset_config['path']

        if not os.path.exists(dataset_path):
            print(f"Error: Dataset file not found: {dataset_path}")
            print(f"Please place your .txt file at: {dataset_path}")
            sys.exit(1)

        train_model(args.dataset, dataset_path, config, device, args.resume, dataset_config)

    elif args.all:
        # Train on all enabled datasets
        enabled_datasets = [(name, cfg) for name, cfg in config['datasets'].items()
                          if cfg.get('enabled', False)]

        if not enabled_datasets:
            print("Error: No datasets are enabled in config.yaml")
            print("Set 'enabled: true' for at least one dataset")
            sys.exit(1)

        print(f"Training on {len(enabled_datasets)} datasets sequentially...")

        for dataset_name, dataset_config in enabled_datasets:
            dataset_path = dataset_config['path']

            if not os.path.exists(dataset_path):
                print(f"Warning: Skipping {dataset_name} - file not found: {dataset_path}")
                continue

            train_model(dataset_name, dataset_path, config, device, dataset_config=dataset_config)

    else:
        print("Error: Please specify --dataset <name> or --all")
        print("Use --help for more information")
        sys.exit(1)


if __name__ == '__main__':
    main()
