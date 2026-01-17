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
import base64

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

def export_vocab(dataset_name):
    """Export vocabulary to JSON format (loads from .pkl or .json)."""

    # Try pkl first, then json
    pkl_path = f'vocab/{dataset_name}_vocab.pkl'
    json_path = f'vocab/{dataset_name}_vocab.json'

    if os.path.exists(pkl_path):
        print(f"\nLoading vocabulary from {pkl_path}...")
        with open(pkl_path, 'rb') as f:
            vocab_meta = pickle.load(f)
    elif os.path.exists(json_path):
        print(f"\nLoading vocabulary from {json_path}...")
        with open(json_path, 'r') as f:
            vocab_meta = json.load(f)
    else:
        raise FileNotFoundError(f"No vocabulary found at {pkl_path} or {json_path}")

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

def create_base64_embeddings():
    """Create base64-encoded model and vocab for HTML embedding."""

    # Export model to ONNX
    model_path, config = export_model_to_onnx()

    # Export vocabulary
    vocab_path, vocab_data = export_vocab()

    print("\n" + "="*60)
    print("Creating base64 embeddings for HTML...")
    print("="*60)

    # Read and encode model
    with open(model_path, 'rb') as f:
        model_bytes = f.read()

    model_base64 = base64.b64encode(model_bytes).decode('utf-8')
    model_size_mb = len(model_bytes) / (1024 * 1024)

    print(f"\nModel: {model_size_mb:.2f} MB")
    print(f"Base64 encoded size: {len(model_base64) / (1024 * 1024):.2f} MB")

    # Read and encode vocab
    with open(vocab_path, 'r') as f:
        vocab_json = f.read()

    vocab_size_kb = len(vocab_json) / 1024
    print(f"Vocabulary: {vocab_size_kb:.2f} KB")

    # Create JavaScript snippet for embedding
    js_snippet = f'''
// ============================================================================
// EMBEDDED MODEL AND VOCABULARY
// ============================================================================

// Model configuration
const MODEL_CONFIG = {{
    block_size: {config.block_size},
    vocab_size: {config.vocab_size},
    n_layer: {config.n_layer},
    n_head: {config.n_head},
    n_embd: {config.n_embd},
    cond_dim: {config.cond_dim}
}};

// Vocabulary mapping
const VOCAB = {vocab_json};

// ONNX model as base64
// Size: {model_size_mb:.2f} MB
// Note: In production, split this into chunks if needed
const MODEL_BASE64 = '{model_base64[:100]}...'; // Truncated for display

// Full model data (uncomment for production):
// const MODEL_BASE64 = '{model_base64}';

// Helper function to decode base64 to Uint8Array
function base64ToUint8Array(base64) {{
    const binaryString = atob(base64);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {{
        bytes[i] = binaryString.charCodeAt(i);
    }}
    return bytes;
}}

// Load model when needed
async function loadModel() {{
    const modelBytes = base64ToUint8Array(MODEL_BASE64);
    // Will be used by ONNX Runtime Web
    return modelBytes;
}}
'''

    # Save the snippet
    output_path = 'model_embedding.js'
    with open(output_path, 'w') as f:
        f.write(js_snippet)

    print(f"\nJavaScript embedding saved to {output_path}")

    # Also save the full base64 to a separate file
    full_base64_path = 'models/model_base64.txt'
    with open(full_base64_path, 'w') as f:
        f.write(model_base64)

    print(f"Full base64 saved to {full_base64_path}")

    # Calculate total HTML size estimate
    total_size = model_size_mb + (vocab_size_kb / 1024)
    print("\n" + "="*60)
    print("SIZE ESTIMATES FOR HTML FILE")
    print("="*60)
    print(f"Model (FP32):                {model_size_mb:.2f} MB")
    print(f"Vocabulary:                  {vocab_size_kb:.2f} KB")
    print(f"ONNX Runtime Web (~2.5MB):   2.50 MB")
    print(f"JavaScript code (~50KB):     0.05 MB")
    print(f"Shaders & Audio (~20KB):     0.02 MB")
    print(f"Font embedded (~15KB):       0.01 MB")
    print("-" * 60)
    print(f"TOTAL ESTIMATED SIZE:        {total_size + 2.58:.2f} MB")
    print("="*60)

    if total_size + 2.58 < 100:
        print("[OK] Well within 100MB budget!")
    else:
        print("[WARNING] Exceeds 100MB, may need further optimization")

    return {
        'model_path': model_path,
        'model_base64_path': full_base64_path,
        'vocab_path': vocab_path,
        'config': config,
        'vocab_data': vocab_data
    }

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

    # Export model to ONNX (output will be {output}_quantized.onnx if quantization succeeds)
    final_model_path, config = export_model_to_onnx(args.model, args.output)

    print("\n" + "="*60)
    print("EXPORT COMPLETE!")
    print("="*60)
    print(f"ONNX model: {final_model_path}")
    print(f"Vocabulary: {args.vocab}")
    print("\nNext: Run update_model.py to generate we.html")
    print(f"  python scripts/art-piece/update_model.py --model {final_model_path} --vocab {args.vocab}")
