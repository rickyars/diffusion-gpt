"""Quick script to inspect the pretrained model checkpoint."""
import torch

checkpoint = torch.load('pretrained_model/model_epoch_25.pth', map_location='cpu', weights_only=False)
print("Checkpoint keys:")
for key in checkpoint.keys():
    print(f"  {key}")

if 'vocab_size' in checkpoint:
    print(f"\nVocab size: {checkpoint['vocab_size']}")

if 'itos' in checkpoint:
    print(f"\nitos available: {len(checkpoint['itos'])} characters")
    print("First 20 chars:", list(checkpoint['itos'].items())[:20])

if 'stoi' in checkpoint:
    print(f"\nstoi available: {len(checkpoint['stoi'])} characters")

if 'config' in checkpoint:
    print("\nModel config:")
    for k, v in checkpoint['config'].items():
        print(f"  {k}: {v}")
