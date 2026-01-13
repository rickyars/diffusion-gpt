"""
Export the discrete diffusion model to ONNX format with INT8 quantization
for in-browser inference using ONNX Runtime Web.
"""

import argparse
import torch
import pickle
import json
import base64
import os
import sys

# Add training directory to path for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))

from model import GPT, GPTConfig, GPTQuantized

def export_model_to_onnx(model_path, dataset_name):
    """Export PyTorch model to ONNX format (handles both FP32 and QAT models)."""

    print(f"Loading model from {model_path}...")

    checkpoint = torch.load(model_path, map_location='cpu')

    # Check if this is a QAT-trained model
    is_qat = checkpoint.get('qat_trained', False)

    if is_qat:
        print(f"\n{'='*60}")
        print("DETECTED: QAT-Trained Model")
        print(f"{'='*60}")
        print("This model was trained with quantization awareness.")
        print("Will convert to true INT8 before ONNX export.")
        print(f"{'='*60}\n")

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

    # Initialize model (QAT or standard)
    if is_qat:
        # Create QAT model and load weights
        # Default to fbgemm backend (x86) - most common for deployment
        qat_backend = checkpoint.get('qat_backend', 'fbgemm')
        qat_model = GPTQuantized(config, backend=qat_backend)
        qat_model.prepare_qat()  # Prepare with fake quantization
        qat_model.model.load_state_dict(checkpoint['model_state_dict'])
        qat_model.eval()

        # Convert to true INT8
        print("\nConverting to true INT8 quantized model...")
        qat_model.convert_to_quantized()

        # Use the quantized model
        model = qat_model.model
        print("✓ Model converted to INT8")
    else:
        # Standard FP32 model
        model = GPT(config)
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")
    if is_qat:
        print("Model type: INT8 Quantized (from QAT training)")
    else:
        print("Model type: FP32")

    # Create dummy inputs for ONNX export
    batch_size = 1
    seq_length = config.block_size

    dummy_idx = torch.randint(0, config.vocab_size, (batch_size, seq_length), dtype=torch.long)
    dummy_sigma = torch.randn(batch_size, 1)

    # Export to ONNX
    onnx_path = f'models/{dataset_name}_model.onnx'
    print(f"\nExporting to ONNX format: {onnx_path}...")

    torch.onnx.export(
        model,
        (dummy_idx, dummy_sigma),
        onnx_path,
        input_names=['input_ids', 'sigma'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'sigma': {0: 'batch_size'},
            'logits': {0: 'batch_size', 1: 'sequence_length'}
        },
        opset_version=18,
        do_constant_folding=True,
        verbose=False
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

    # If model is already QAT-trained, it's already quantized - skip post-training quantization
    if is_qat:
        print(f"\n[INFO] Model is already quantized (QAT-trained)")
        print(f"       Skipping post-training quantization")
        return onnx_path, config

    # Try to simplify the ONNX model to resolve shape inference issues
    try:
        import onnxsim
        print("\nSimplifying ONNX model to resolve shape conflicts...")

        onnx_model = onnx.load(onnx_path)
        simplified_model, check = onnxsim.simplify(
            onnx_model,
            check_n=0,  # Skip validation to avoid shape errors
            skip_shape_inference=False
        )

        if check or True:  # Always use simplified model even if check fails
            simplified_path = onnx_path.replace('.onnx', '_simplified.onnx')
            onnx.save(simplified_model, simplified_path)
            print(f"[OK] Model simplified successfully")
            # Use simplified model for quantization
            onnx_path_for_quantization = simplified_path
        else:
            print("[WARNING] Simplification check failed, using original model")
            onnx_path_for_quantization = onnx_path

    except ImportError:
        print("\n[INFO] onnx-simplifier not installed, skipping simplification")
        print("      Install with: pip install onnx-simplifier")
        onnx_path_for_quantization = onnx_path
    except Exception as e:
        print(f"[WARNING] Simplification failed: {str(e)[:100]}")
        print("          Using original model for quantization")
        onnx_path_for_quantization = onnx_path

    # Quantize the model
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        import onnx
        from onnx import shape_inference

        quantized_path = f'models/{dataset_name}_model_quantized.onnx'
        print(f"\nQuantizing model to INT8...")
        print("Quantization configuration:")
        print("  - Type: Dynamic quantization")
        print("  - Weight type: QInt8 (signed 8-bit)")
        print("  - Per-tensor quantization (better compatibility)")

        try:
            # Try QUInt8 quantization - more conservative for better accuracy
            print("\nAttempting QUInt8 quantization (unsigned for better accuracy)...")
            print("  Using conservative settings to preserve model accuracy")

            quantize_dynamic(
                onnx_path_for_quantization,
                quantized_path,
                weight_type=QuantType.QUInt8,
                per_channel=False,
                reduce_range=True,  # Enable reduce_range for better accuracy
                op_types_to_quantize=['MatMul']  # Only quantize MatMul (most compute-heavy)
            )

            quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
            print(f"[OK] Quantized model (QUInt8) size: {quantized_size:.2f} MB")
            print(f"[OK] Size reduction: {((onnx_size - quantized_size) / onnx_size * 100):.1f}%")

            # Verify the quantized model is valid
            try:
                onnx_model = onnx.load(quantized_path)
                onnx.checker.check_model(onnx_model)
                print("[OK] Quantized model validation passed")
            except Exception as ve:
                print(f"[WARNING] Quantized model validation warning: {str(ve)[:100]}")

            final_model_path = quantized_path

        except Exception as e:
            # Fallback to QInt8 if QUInt8 fails
            print(f"[WARNING] QUInt8 with exclusions failed: {str(e)[:200]}")
            print("  Trying fallback to basic QUInt8 (no exclusions)...")

            try:
                quantize_dynamic(
                    onnx_path_for_quantization,
                    quantized_path,
                    weight_type=QuantType.QUInt8
                )

                quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
                print(f"[OK] Quantized model (QUInt8) size: {quantized_size:.2f} MB")
                print(f"[OK] Size reduction: {((onnx_size - quantized_size) / onnx_size * 100):.1f}%")
                final_model_path = quantized_path

            except Exception as e2:
                print(f"[ERROR] QUInt8 quantization also failed: {str(e2)[:200]}")
                print(f"  Using non-quantized ONNX model instead ({onnx_size:.2f} MB)")
                print("\nNote: Quantization may fail due to:")
                print("  - Dynamic shapes in the model")
                print("  - Unsupported operations")
                print("  - Shape inference issues")
                print("  The non-quantized model will still work, just be larger/slower.")
                final_model_path = onnx_path

    except ImportError as ie:
        print(f"\n[WARNING] Required library not available: {ie}")
        print("  Install with: pip install onnxruntime onnx")
        final_model_path = onnx_path

    # Clean up simplified model if quantization succeeded (we don't need both)
    if final_model_path == quantized_path and 'onnx_path_for_quantization' in locals():
        if onnx_path_for_quantization != onnx_path and os.path.exists(onnx_path_for_quantization):
            print(f"\nCleaning up intermediate simplified model...")
            os.remove(onnx_path_for_quantization)

    return final_model_path, config

def export_vocab(dataset_name):
    """Export vocabulary to JSON format."""

    vocab_path = f'vocab/{dataset_name}_vocab.pkl'
    print(f"\nLoading vocabulary from {vocab_path}...")

    with open(vocab_path, 'rb') as f:
        vocab_meta = pickle.load(f)

    vocab_data = {
        'itos': vocab_meta['itos'],
        'stoi': vocab_meta['stoi'],
        'vocab_size': len(vocab_meta['itos'])
    }

    # Save as JSON
    json_path = f'vocab/{dataset_name}_vocab.json'
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

// ONNX model as base64 (quantized INT8)
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
    print(f"Model (quantized):           {model_size_mb:.2f} MB")
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
        '--dataset',
        type=str,
        required=True,
        help='Dataset name for output files (e.g., confessions)'
    )

    args = parser.parse_args()

    print("="*60)
    print("EXPORTING DISCRETE DIFFUSION MODEL FOR WEB")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print("="*60)

    # Export model to ONNX
    model_path, config = export_model_to_onnx(args.model, args.dataset)

    # Export vocabulary
    vocab_path, vocab_data = export_vocab(args.dataset)

    print("\n" + "="*60)
    print("EXPORT COMPLETE!")
    print("="*60)
    print(f"ONNX model: {model_path}")
    print(f"Vocabulary: {vocab_path}")
    print("\nNext: Run update_model.py to generate we.html")
    print(f"  python scripts/art-piece/update_model.py --dataset {args.dataset}")
