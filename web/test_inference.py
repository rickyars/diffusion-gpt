"""
Test script to compare Python vs ONNX inference.
This will help debug the gibberish output issue.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
import numpy as np
from model import GPT, GPTConfig

# Load the PyTorch model
checkpoint_path = '../pretrained_model/model_epoch_25.pth'
print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Extract model config
model_config_dict = checkpoint['config']
model_config = GPTConfig(**model_config_dict)
vocab_size = checkpoint.get('vocab_size', model_config.vocab_size)

print(f"Vocab size: {vocab_size}")
print(f"Block size: {model_config.block_size}")

# Initialize model
model = GPT(model_config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Get vocabulary
if 'itos' in checkpoint and 'stoi' in checkpoint:
    itos = checkpoint['itos']
    stoi = checkpoint['stoi']
    print(f"Vocabulary loaded from checkpoint")
else:
    print("Warning: No vocab in checkpoint, using default")
    vocab_str = "\n !\"&'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    itos = {i: c for i, c in enumerate(vocab_str)}
    stoi = {c: i for i, c in enumerate(vocab_str)}

print(f"\nVocabulary sample (first 10 chars):")
for i in range(min(10, len(itos))):
    char = itos[i]
    display = repr(char) if char in ['\n', ' '] else char
    print(f"  {i}: {display}")

# Test input
test_input = torch.randint(0, vocab_size, (1, model_config.block_size))
test_sigma = torch.tensor([1.0])

print(f"\nTest input shape: {test_input.shape}")
print(f"Test sigma: {test_sigma.item()}")

# Run PyTorch inference
with torch.no_grad():
    output = model(test_input, test_sigma)

print(f"Output shape: {output.shape}")
print(f"Output min: {output.min().item():.4f}")
print(f"Output max: {output.max().item():.4f}")
print(f"Output mean: {output.mean().item():.4f}")

# Save test data for JavaScript comparison
test_data = {
    'input_ids': test_input[0].tolist(),
    'sigma': test_sigma.item(),
    'output_logits': output[0].tolist(),  # Shape: [seq_len, vocab_size]
    'vocab_size': vocab_size,
    'block_size': model_config.block_size
}

with open('test_data.json', 'w') as f:
    json.dump(test_data, f, indent=2)
print("\nSaved test data to test_data.json")

# Also save the full config
full_config = {
    'vocab_size': vocab_size,
    'block_size': model_config.block_size,
    'n_layer': model_config.n_layer,
    'n_head': model_config.n_head,
    'n_embd': model_config.n_embd,
    'cond_dim': model_config.cond_dim,
}

with open('test_config.json', 'w') as f:
    json.dump(full_config, f, indent=2)
print("Saved config to test_config.json")

# Test the geometric noise calculation
sigma_min = 0.0001
sigma_max = 20.0

print(f"\nGeometric noise test (sigma_min={sigma_min}, sigma_max={sigma_max}):")
test_t_values = [1.0, 0.5, 0.0001]
for t in test_t_values:
    bar_sigma = sigma_min ** (1 - t) * sigma_max ** t
    print(f"  t={t:.4f} -> bar_sigma={bar_sigma:.6f}")

# Test one denoising step
print("\n" + "="*60)
print("Testing one complete denoising step:")
print("="*60)

from utils import geometricNoise, staggered_score, transition, sample_categorical

# Initialize random sequence
x = torch.randint(0, vocab_size, (1, model_config.block_size))
print(f"\nInitial random text:")
print(''.join([itos[int(i)] for i in x[0][:100]]))  # First 100 chars

# Denoising parameters
steps = 64
eps = 1e-5
current_step = 0

# Calculate timestep
t = 1 - (current_step / steps) * (1 - eps)
t_tensor = torch.tensor([[t]])

# Geometric noise
sigma_min_tensor = torch.tensor(sigma_min)
sigma_max_tensor = torch.tensor(sigma_max)
curr_sigma_bar = sigma_min_tensor ** (1 - t) * sigma_max_tensor ** t

# Next sigma
t_next = 1 - ((current_step + 1) / steps) * (1 - eps)
next_sigma_bar = sigma_min_tensor ** (1 - t_next) * sigma_max_tensor ** t_next
delta_sigma = curr_sigma_bar - next_sigma_bar

print(f"\nStep {current_step}:")
print(f"  t = {t:.6f}")
print(f"  curr_sigma_bar = {curr_sigma_bar:.6f}")
print(f"  next_sigma_bar = {next_sigma_bar:.6f}")
print(f"  delta_sigma = {delta_sigma:.6f}")

# Run model
with torch.no_grad():
    log_score = model(x, curr_sigma_bar.unsqueeze(0))
    score = torch.exp(log_score)

print(f"  log_score shape: {log_score.shape}")
print(f"  log_score range: [{log_score.min():.4f}, {log_score.max():.4f}]")
print(f"  score range: [{score.min():.4f}, {score.max():.4f}]")

# Apply staggered score
stag_score = staggered_score(score, delta_sigma)
print(f"  stag_score range: [{stag_score.min():.4f}, {stag_score.max():.4f}]")

# Get transition
trans = transition(x, delta_sigma, vocab_size)
print(f"  trans shape: {trans.shape}")
print(f"  trans range: [{trans.min():.4f}, {trans.max():.4f}]")

# Compute probs
probs = stag_score * trans
print(f"  probs shape: {probs.shape}")
print(f"  probs range: [{probs.min():.4f}, {probs.max():.4f}]")
print(f"  probs sum (first position): {probs[0, 0].sum():.6f}")

# Sample
x_new = sample_categorical(probs)
print(f"\nAfter one denoising step:")
print(''.join([itos[int(i)] for i in x_new[0][:100]]))  # First 100 chars

# Count how many tokens changed
num_changed = (x != x_new).sum().item()
total_tokens = x.numel()
print(f"\nTokens changed: {num_changed}/{total_tokens} ({100*num_changed/total_tokens:.1f}%)")

print("\n" + "="*60)
print("Diagnostic complete!")
print("="*60)
