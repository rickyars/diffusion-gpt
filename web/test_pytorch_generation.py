"""
Quick test to verify PyTorch model generates coherent text.
This will tell us if the checkpoint is actually trained.

USAGE:
    python test_pytorch_generation.py --model ../models/shakespeare.pt
"""
import sys
import os
import argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model import GPT, GPTConfig
from utils import GeometricNoise, staggered_score, transition, sample_categorical, decode

# Parse arguments
parser = argparse.ArgumentParser(description='Test PyTorch model generation')
parser.add_argument('--model', type=str, default='../pretrained_model/model_epoch_25.pth',
                    help='Path to model checkpoint (e.g., ../models/shakespeare.pt)')
args = parser.parse_args()

print("="*70)
print("TESTING PYTORCH MODEL GENERATION")
print("="*70)

# Load checkpoint
checkpoint_path = args.model
if not os.path.exists(checkpoint_path):
    print(f"\n❌ ERROR: Model checkpoint not found: {checkpoint_path}")
    print("\nUsage: python test_pytorch_generation.py --model ../models/shakespeare.pt")
    sys.exit(1)

print(f"\nLoading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

# Extract state dict - handle different checkpoint formats
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    print("Found 'model_state_dict' in checkpoint")
elif 'model' in checkpoint:
    state_dict = checkpoint['model']
    print("Found 'model' in checkpoint")
else:
    # Assume the checkpoint IS the state dict
    state_dict = checkpoint
    print("Using checkpoint as state dict")

# Infer config from state dict
vocab_size = state_dict['transformer.wte.weight'].shape[0]
n_embd = state_dict['transformer.wte.weight'].shape[1]
block_size = state_dict['transformer.wpe.weight'].shape[0]
cond_dim = state_dict['sigma_map.mlp.0.weight'].shape[0]

# Count layers
n_layer = 0
for key in state_dict.keys():
    if key.startswith('transformer.h.'):
        layer_num = int(key.split('.')[2])
        n_layer = max(n_layer, layer_num + 1)

config = GPTConfig(
    vocab_size=vocab_size,
    block_size=block_size,
    n_layer=n_layer,
    n_head=6,
    n_embd=n_embd,
    cond_dim=cond_dim,
    dropout=0.0,
    bias=False
)

print(f"Config: vocab_size={vocab_size}, block_size={block_size}, n_layer={n_layer}, cond_dim={cond_dim}")

# Initialize model
model = GPT(config)
model.load_state_dict(state_dict)
model.eval()

# Get vocabulary
if 'itos' in checkpoint and 'stoi' in checkpoint:
    itos = checkpoint['itos']
    stoi = checkpoint['stoi']
    print("Vocabulary loaded from checkpoint")
else:
    # Use default Shakespeare vocabulary
    print("Using default vocabulary")
    vocab_str = "\n !\"&'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    itos = {i: c for i, c in enumerate(vocab_str)}
    stoi = {c: i for i, c in enumerate(vocab_str)}

# Generate sample
print("\n" + "="*70)
print("GENERATING TEXT (10 steps for quick test)")
print("="*70)

torch.manual_seed(42)
x = torch.randint(0, vocab_size, (1, 256))
noise_schedule = GeometricNoise(sigma_min=0.0001, sigma_max=20.0)

print("\nInitial random text:")
print(decode(x[0][:100], itos))

steps = 10  # Quick test with just 10 steps
eps = 1e-5

with torch.no_grad():
    for i in range(steps + 1):
        t = 1 - (i / steps) * (1 - eps)
        t_tensor = torch.tensor([[t]])

        curr_sigma_bar, _ = noise_schedule(t_tensor)

        if i < steps:
            t_next = 1 - ((i + 1) / steps) * (1 - eps)
            t_next_tensor = torch.tensor([[t_next]])
            next_sigma_bar, _ = noise_schedule(t_next_tensor)
            delta_sigma = curr_sigma_bar - next_sigma_bar

            log_score = model(x, curr_sigma_bar)
            score = torch.exp(log_score)

            stag_score = staggered_score(score, delta_sigma)
            probs = stag_score * transition(x, delta_sigma, vocab_size)
            x = sample_categorical(probs)

        if i % 2 == 0:  # Print every 2 steps
            print(f"\nStep {i}/{steps} (sigma={curr_sigma_bar.item():.4f}):")
            print(decode(x[0][:100], itos))

print("\n" + "="*70)
print("FINAL TEXT (first 200 chars):")
print("="*70)
print(decode(x[0][:200], itos))

print("\n" + "="*70)
print("DIAGNOSIS:")
print("="*70)
final_text = decode(x[0], itos)
if final_text.count(' ') > 10 and any(word in final_text.lower() for word in ['the', 'and', 'to', 'of', 'a']):
    print("✅ TEXT LOOKS COHERENT - Model is working!")
    print("   The model is properly trained.")
    print("   Issue must be in ONNX export or web inference.")
else:
    print("❌ TEXT IS GIBBERISH - Model may not be trained")
    print("   Or 10 steps is too few for this model.")
    print("   Try running full generation with generate.py")
