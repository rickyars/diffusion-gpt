#!/usr/bin/env python3
"""
Diagnostic script to identify training performance issues.
Run this before training to check for potential bottlenecks.
"""

import os
import sys
import yaml
import torch

def check_gpu():
    """Check GPU availability and memory."""
    print("=" * 80)
    print("GPU CHECK")
    print("=" * 80)

    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    else:
        print("✗ CUDA not available - training will be VERY slow on CPU!")
    print()

def check_datasets(config):
    """Check dataset sizes and validate paths."""
    print("=" * 80)
    print("DATASET CHECK")
    print("=" * 80)

    enabled_datasets = [(name, cfg) for name, cfg in config['datasets'].items()
                       if cfg.get('enabled', False)]

    for name, dataset_config in enabled_datasets:
        path = dataset_config['path']
        max_chars = dataset_config.get('max_chars', None)

        if not os.path.exists(path):
            print(f"✗ {name}: FILE NOT FOUND - {path}")
            continue

        # Check actual file size
        file_size = os.path.getsize(path)
        with open(path, 'r', encoding='utf-8') as f:
            actual_chars = len(f.read())

        effective_chars = min(actual_chars, max_chars) if max_chars else actual_chars

        print(f"✓ {name}:")
        print(f"    File: {path}")
        print(f"    File size: {file_size / 1e6:.1f} MB")
        print(f"    Total characters: {actual_chars:,}")
        if max_chars:
            print(f"    Limit: {max_chars:,} chars")
            print(f"    Effective size: {effective_chars:,} chars ({effective_chars/actual_chars*100:.1f}%)")
            if effective_chars < max_chars:
                print(f"    ⚠ WARNING: File is smaller than max_chars limit!")
        else:
            print(f"    ⚠ WARNING: No max_chars limit! Training on full dataset ({actual_chars:,} chars)")
        print()

def estimate_training_time(config, dataset_chars):
    """Estimate training time based on configuration."""
    print("=" * 80)
    print("TRAINING TIME ESTIMATE")
    print("=" * 80)

    batch_size = config['training']['batch_size']
    context_length = config['model']['context_length']
    epochs = config['training']['epochs']
    val_split = config['training']['val_split']

    # Calculate training set size
    train_chars = int(dataset_chars * (1 - val_split))
    num_samples = max(0, train_chars - context_length)
    batches_per_epoch = num_samples // batch_size
    total_batches = batches_per_epoch * epochs

    print(f"Dataset characters: {dataset_chars:,}")
    print(f"Training samples: {num_samples:,}")
    print(f"Batches per epoch: {batches_per_epoch:,}")
    print(f"Total batches: {total_batches:,}")
    print()

    # Rough estimates (adjust based on your hardware)
    # RTX 4080 Super typically processes ~500-2000 samples/sec depending on model size
    estimated_samples_per_sec = 1000  # Conservative estimate
    estimated_time_hours = (total_batches * batch_size) / estimated_samples_per_sec / 3600

    print(f"Estimated time at {estimated_samples_per_sec} samples/sec: {estimated_time_hours:.1f} hours")
    print(f"  (This is a rough estimate - actual speed varies)")
    print()

def check_config_performance(config):
    """Check for performance-impacting configuration."""
    print("=" * 80)
    print("CONFIGURATION ANALYSIS")
    print("=" * 80)

    issues = []
    recommendations = []

    # Check batch size
    batch_size = config['training']['batch_size']
    if batch_size > 128:
        issues.append(f"⚠ Batch size is large ({batch_size}) - may cause GPU memory issues")
        recommendations.append("  → Try batch_size: 64 or 128 for 4080 Super (16GB VRAM)")
    elif batch_size < 64:
        recommendations.append(f"  Consider increasing batch size to 64-128 for faster training")
    else:
        print(f"✓ Batch size: {batch_size} (good for 16GB GPU)")

    # Check torch.compile
    use_compile = config['training'].get('use_compile', False)
    if not use_compile:
        issues.append("⚠ torch.compile is disabled")
        recommendations.append("  → Set use_compile: true for 30-50% speedup (PyTorch 2.0+)")
    else:
        print("✓ torch.compile enabled (30-50% faster training)")

    # Check epochs
    epochs = config['training']['epochs']
    if epochs > 50:
        issues.append(f"⚠ Training for {epochs} epochs may be overkill")
        recommendations.append("  → Most models converge by 25-50 epochs")

    print()
    if issues:
        print("Issues found:")
        for issue in issues:
            print(issue)
        print()

    if recommendations:
        print("Recommendations:")
        for rec in recommendations:
            print(rec)
        print()

def main():
    config_path = 'config.yaml'

    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    check_gpu()
    check_datasets(config)
    check_config_performance(config)

    # Estimate for first enabled dataset
    enabled_datasets = [(name, cfg) for name, cfg in config['datasets'].items()
                       if cfg.get('enabled', False)]
    if enabled_datasets:
        name, dataset_config = enabled_datasets[0]
        path = dataset_config['path']
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                actual_chars = len(f.read())
            max_chars = dataset_config.get('max_chars', None)
            effective_chars = min(actual_chars, max_chars) if max_chars else actual_chars
            estimate_training_time(config, effective_chars)

    print("=" * 80)
    print("Run this script again after making config changes to verify improvements.")
    print("=" * 80)

if __name__ == '__main__':
    main()
