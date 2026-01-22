"""
Export the discrete diffusion model to ONNX format
for in-browser inference using ONNX Runtime Web.
"""

import argparse
import os
import sys

import torch
import pickle
import json

# Add training directory to path for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))

from model import GPT, GPTConfig

def export_model_to_onnx(model_path, output_path):
    """Export PyTorch model to ONNX format."""

    print(f"Loading model from {model_path}...")

    checkpoint = torch.load(model_path, map_location='cpu')

    # Extract config
    if 'config' in checkpoint:
        config = GPTConfig(**checkpoint['config'])
    else:
        # Fallback to default config
        config = GPTConfig(
            block_size=256,
            vocab_size=checkpoint.get('vocab_size', 65),
            n_layer=6,
            n_head=6,
            n_embd=384,
            cond_dim=64,
            dropout=0.0,  # No dropout for inference
            bias=False
        )

    print(f"Model config: {config}")

    # Initialize model
    model = GPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
    print("Model type: FP32")

    # Create dummy inputs for ONNX export
    batch_size = 1
    seq_length = config.block_size

    dummy_idx = torch.randint(0, config.vocab_size, (batch_size, seq_length), dtype=torch.long)
    dummy_sigma = torch.randn(batch_size, 1)

    # Export to ONNX
    onnx_path = output_path
    print(f"\nExporting to ONNX format: {onnx_path}...")

    # Export with static shapes - dynamic shapes cause issues with onnxruntime quantization
    # The browser inference always uses fixed block_size anyway
    torch.onnx.export(
        model,
        (dummy_idx, dummy_sigma),
        onnx_path,
        input_names=['input_ids', 'sigma'],
        output_names=['logits'],
        opset_version=18,
        do_constant_folding=True,
        verbose=False,
        dynamo=True  # Use torch.export-based ONNX exporter (PyTorch 2.9+)
    )

    # Check if external data was created and merge it
    data_path = f"{onnx_path}.data"
    if os.path.exists(data_path):
        print(f"\nEmbedding external data from {data_path}...")
        try:
            import onnx
            onnx_model = onnx.load(onnx_path)
            onnx.save(onnx_model, onnx_path)
            # Remove external data file since it's now embedded
            os.remove(data_path)
            print("[OK] External data embedded and merged")
        except Exception as e:
            print(f"Note: External data file exists. Run merge_onnx_data.py to merge: {e}")

    # Check ONNX file size
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"ONNX model size: {onnx_size:.2f} MB")

    return onnx_path, config

def export_vocab(vocab_path):
    """Export vocabulary to JSON format (loads from .pkl or .json)."""

    # Determine file type and output path
    if vocab_path.endswith('.pkl'):
        json_path = vocab_path.replace('.pkl', '.json')
        print(f"\nLoading vocabulary from {vocab_path}...")
        with open(vocab_path, 'rb') as f:
            vocab_meta = pickle.load(f)
    elif vocab_path.endswith('.json'):
        json_path = vocab_path
        print(f"\nLoading vocabulary from {vocab_path}...")
        with open(vocab_path, 'r') as f:
            vocab_meta = json.load(f)
    else:
        raise ValueError(f"Vocabulary file must be .pkl or .json, got: {vocab_path}")

    vocab_data = {
        'itos': vocab_meta['itos'],
        'stoi': vocab_meta['stoi'],
        'vocab_size': len(vocab_meta['itos'])
    }

    # Save as JSON (update if loaded from pkl)
    with open(json_path, 'w') as f:
        json.dump(vocab_data, f, separators=(',', ':'))

    json_size = os.path.getsize(json_path) / 1024
    print(f"Vocabulary exported to {json_path} ({json_size:.2f} KB)")

    return json_path, vocab_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export PyTorch discrete diffusion model to ONNX format for web deployment'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to PyTorch model checkpoint (e.g., models/confessions_epoch_15.pt)'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        required=True,
        help='Path to vocabulary file (e.g., vocab/confessions_vocab.pkl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/model.onnx',
        help='Output path for ONNX model (default: models/model.onnx)'
    )

    args = parser.parse_args()

    print("="*60)
    print("EXPORTING DISCRETE DIFFUSION MODEL FOR WEB")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Vocab: {args.vocab}")
    print(f"Output: {args.output}")
    print("="*60)

    # Export model to ONNX
    final_model_path, config = export_model_to_onnx(args.model, args.output)

    # Export vocabulary to JSON
    vocab_json_path, vocab_data = export_vocab(args.vocab)

    print("\n" + "="*60)
    print("EXPORT COMPLETE!")
    print("="*60)
    print(f"ONNX model: {final_model_path}")
    print(f"Vocabulary: {vocab_json_path}")
    print("\nNext: Run update_model.py to generate we.html")
    print(f"  python scripts/art-piece/update_model.py --model {final_model_path} --vocab {vocab_json_path}")
