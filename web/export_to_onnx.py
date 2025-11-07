"""
Export the trained PyTorch model to ONNX format for browser inference.

USAGE:
    cd web
    python export_to_onnx.py

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

    # Extract model config
    if 'config' in checkpoint:
        model_config_dict = checkpoint['config']
        model_config = GPTConfig(**model_config_dict)
        print(f"Model config loaded from checkpoint")
    else:
        raise ValueError("Model config not found in checkpoint")

    vocab_size = checkpoint.get('vocab_size', model_config.vocab_size)
    print(f"Vocab size: {vocab_size}")

    # Initialize model
    print("Initializing model...")
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
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
    export_model_to_onnx()
