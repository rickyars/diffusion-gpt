"""
Export the trained PyTorch model to ONNX format for browser inference.

USAGE:
    cd web
    python export_to_onnx.py --checkpoint ../models/shakespeare.pt

Options:
    --checkpoint PATH    Path to model checkpoint (default: ../pretrained_model/model_epoch_25.pth)
    --output DIR         Output directory for ONNX files (default: ./demo)
    --opset VERSION      ONNX opset version (default: 15)

This will create a demo/ directory with:
    - model.onnx (~45MB) - ONNX format model
    - vocab.json (<1KB) - Character vocabulary
    - metadata.json (<1KB) - Model configuration
"""
import os
import sys
import json
import pickle
import argparse

# Add parent directory to path to import model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from model import GPT, GPTConfig

def export_model_to_onnx(
    checkpoint_path: str = '../pretrained_model/model_epoch_25.pth',
    output_dir: str = './demo',
    opset_version: int = 15
):
    """
    Export trained model to ONNX format.

    Args:
        checkpoint_path: Path to PyTorch checkpoint
        output_dir: Directory to save ONNX model and metadata
        opset_version: ONNX opset version
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Inspect checkpoint structure
    print(f"Checkpoint keys: {list(checkpoint.keys())[:10]}...")  # Show first 10 keys

    # Extract model config - handle different checkpoint formats
    if 'config' in checkpoint:
        print("Found 'config' in checkpoint")
        model_config_dict = checkpoint['config']
        model_config = GPTConfig(**model_config_dict)
    elif 'model_args' in checkpoint:
        print("Found 'model_args' in checkpoint")
        model_config_dict = checkpoint['model_args']
        model_config = GPTConfig(**model_config_dict)
    else:
        print("⚠️  No config found in checkpoint, inferring from state_dict...")
        # Try to infer config from model structure
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))

        # Extract dimensions from the state dict
        if 'transformer.wte.weight' in state_dict:
            vocab_size_inferred = state_dict['transformer.wte.weight'].shape[0]
            n_embd = state_dict['transformer.wte.weight'].shape[1]
        else:
            vocab_size_inferred = 65  # Default Shakespeare vocab
            n_embd = 384  # Default from config

        # Count layers
        layer_count = 0
        for key in state_dict.keys():
            if key.startswith('transformer.h.'):
                layer_num = int(key.split('.')[2])
                layer_count = max(layer_count, layer_num + 1)
        n_layer = layer_count if layer_count > 0 else 6

        # Infer block_size from position embeddings
        if 'transformer.wpe.weight' in state_dict:
            block_size = state_dict['transformer.wpe.weight'].shape[0]
        else:
            block_size = 256  # Default

        # Infer cond_dim from sigma_map layer size
        if 'sigma_map.mlp.0.weight' in state_dict:
            # sigma_map.mlp.0.weight has shape [cond_dim, 256]
            cond_dim = state_dict['sigma_map.mlp.0.weight'].shape[0]
        else:
            cond_dim = 128  # Default

        n_head = 6  # Default, hard to infer reliably

        print(f"  Inferred: vocab_size={vocab_size_inferred}, n_embd={n_embd}, n_layer={n_layer}, "
              f"block_size={block_size}, cond_dim={cond_dim}")

        model_config = GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size_inferred,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            cond_dim=cond_dim,
            dropout=0.0,
            bias=False
        )
        model_config_dict = {
            'block_size': block_size,
            'vocab_size': vocab_size_inferred,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'cond_dim': cond_dim,
            'dropout': 0.0,
            'bias': False
        }

    vocab_size = checkpoint.get('vocab_size', model_config.vocab_size)
    print(f"Vocab size: {vocab_size}")

    # Initialize model
    print("Initializing model...")
    model = GPT(model_config)

    # Load state dict - handle different checkpoint structures
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Assume the checkpoint IS the state dict
        model.load_state_dict(checkpoint)

    model.eval()

    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Export to ONNX
    onnx_path = os.path.join(output_dir, 'model.onnx')
    print(f"\nExporting to ONNX: {onnx_path}")

    # Dummy inputs for tracing
    batch_size = 1
    seq_length = model_config.block_size
    dummy_idx = torch.randint(0, vocab_size, (batch_size, seq_length))
    dummy_sigma = torch.tensor([1.0])

    # Export
    torch.onnx.export(
        model,
        (dummy_idx, dummy_sigma),
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input_ids', 'sigma'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size'},
            'sigma': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )

    print(f"ONNX model exported successfully!")

    # Extract and save vocabulary
    if 'itos' in checkpoint and 'stoi' in checkpoint:
        print("\nExtracting vocabulary from checkpoint...")
        vocab_data = {
            'itos': checkpoint['itos'],
            'stoi': checkpoint['stoi']
        }
    else:
        print("\nWarning: Vocabulary not found in checkpoint, using default Shakespeare vocab")
        vocab_str = "\n !\"&'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        vocab_data = {
            'itos': {i: c for i, c in enumerate(vocab_str)},
            'stoi': {c: i for i, c in enumerate(vocab_str)}
        }

    # Save as JSON for JavaScript
    vocab_json_path = os.path.join(output_dir, 'vocab.json')
    with open(vocab_json_path, 'w') as f:
        json.dump(vocab_data, f)
    print(f"Vocabulary saved to: {vocab_json_path}")

    # Save as pickle for Python
    vocab_pkl_path = os.path.join(output_dir, 'vocab.pkl')
    with open(vocab_pkl_path, 'wb') as f:
        pickle.dump(vocab_data, f)
    print(f"Vocabulary saved to: {vocab_pkl_path}")

    # Save model metadata
    metadata = {
        'vocab_size': vocab_size,
        'block_size': model_config.block_size,
        'n_layer': model_config.n_layer,
        'n_head': model_config.n_head,
        'n_embd': model_config.n_embd,
        'cond_dim': model_config.cond_dim,
        'dropout': model_config.dropout,
        'bias': model_config.bias,
        'num_params': model.get_num_params()
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    # Print file sizes
    print("\n" + "="*60)
    print("Export complete! Files created:")
    print("="*60)
    for filename in ['model.onnx', 'vocab.json', 'vocab.pkl', 'metadata.json']:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {filename:20s} {size_mb:8.2f} MB")
    print("="*60)

    return onnx_path, vocab_json_path, metadata_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export PyTorch model to ONNX')
    parser.add_argument('--checkpoint', type=str, default='../pretrained_model/model_epoch_25.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./demo',
                        help='Output directory for ONNX files')
    parser.add_argument('--opset', type=int, default=15,
                        help='ONNX opset version')
    
    args = parser.parse_args()
    
    export_model_to_onnx(
        checkpoint_path=args.checkpoint,
        output_dir=args.output,
        opset_version=args.opset
    )
