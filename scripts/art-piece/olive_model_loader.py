"""
Olive Model Loader for Discrete Diffusion GPT

This script provides the required functions for Olive to load and convert
the custom PyTorch Discrete Diffusion GPT model to ONNX format.

Functions required by Olive:
- _model_loader: Load the PyTorch model from checkpoint
- _io_config: Define input/output configuration for ONNX export
- _dummy_inputs: Generate sample inputs for ONNX conversion
"""

import os
import sys
import torch

# Add training directory to path for model imports
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.join(script_dir, '..', 'training')
sys.path.insert(0, training_dir)

from model import GPT, GPTConfig

# Global config cache - set by _model_loader, used by _dummy_inputs and _io_config
_global_config = None


def _model_loader(model_path: str):
    """
    Load the PyTorch Discrete Diffusion GPT model from checkpoint.

    This function is called by Olive to load the model before conversion.

    Args:
        model_path: Path to the PyTorch checkpoint (.pt file)

    Returns:
        PyTorch model in eval mode
    """
    global _global_config

    print(f"[Olive] Loading model from: {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Extract configuration
    if 'config' in checkpoint:
        config = GPTConfig(**checkpoint['config'])
        _global_config = config  # Cache for _dummy_inputs and _io_config
    else:
        # Fallback configuration if not in checkpoint
        config = GPTConfig(
            block_size=256,
            vocab_size=checkpoint.get('vocab_size', 100),
            n_layer=6,
            n_head=6,
            n_embd=384,
            cond_dim=64,
            dropout=0.0,  # No dropout for inference
            bias=False
        )
        _global_config = config  # Cache for _dummy_inputs and _io_config

    print(f"[Olive] Model config: {config}")

    # Initialize model
    model = GPT(config)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Set to evaluation mode
    model.eval()

    num_params = model.get_num_params() / 1e6
    print(f"[Olive] Model loaded successfully: {num_params:.2f}M parameters")

    return model


def _io_config(model_path):
    """
    Define the input/output configuration for ONNX export.

    This tells Olive about the model's inputs and outputs, including
    their names, types, shapes, and dynamic axes.

    Args:
        model_path: Path to the PyTorch checkpoint (or PyTorchModelHandler from Olive)

    Returns:
        Dictionary with io_config specification
    """
    global _global_config

    # Use cached config from _model_loader if available
    if _global_config is not None:
        block_size = _global_config.block_size
        vocab_size = _global_config.vocab_size
    else:
        # Fallback: extract path if model_path is a model handler object
        if hasattr(model_path, 'model_path'):
            actual_path = model_path.model_path
        else:
            actual_path = model_path

        # Load checkpoint to get configuration
        checkpoint = torch.load(actual_path, map_location='cpu')

        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            block_size = config_dict['block_size']
            vocab_size = config_dict['vocab_size']
        else:
            block_size = 256
            vocab_size = checkpoint.get('vocab_size', 100)

    # Define input/output configuration
    io_config = {
        "input_names": ["input_ids", "sigma"],
        "output_names": ["logits"],
        "input_types": ["int64", "float32"],
        "input_shapes": [
            [1, block_size],  # input_ids: (batch_size, sequence_length)
            [1, 1]            # sigma: (batch_size, 1)
        ],
        "output_types": ["float32"],
        "output_shapes": [
            [1, block_size, vocab_size]  # logits: (batch_size, sequence_length, vocab_size)
        ],
        "dynamic_axes": {
            "input_ids": {
                "0": "batch_size",
                "1": "sequence_length"
            },
            "sigma": {
                "0": "batch_size"
            },
            "logits": {
                "0": "batch_size",
                "1": "sequence_length"
            }
        }
    }

    print(f"[Olive] IO Config:")
    print(f"  - Input: input_ids (int64) [{1}, {block_size}]")
    print(f"  - Input: sigma (float32) [{1}, {1}]")
    print(f"  - Output: logits (float32) [{1}, {block_size}, {vocab_size}]")
    print(f"  - Dynamic axes: batch_size, sequence_length")

    return io_config


def _dummy_inputs(model_path):
    """
    Generate dummy inputs for ONNX conversion and testing.

    This function creates sample inputs that match the model's expected
    input format. Used by Olive during ONNX conversion.

    Args:
        model_path: Path to the PyTorch checkpoint (or PyTorchModelHandler from Olive)

    Returns:
        Tuple of dummy inputs (input_ids, sigma)
    """
    global _global_config

    # Use cached config from _model_loader if available
    if _global_config is not None:
        block_size = _global_config.block_size
        vocab_size = _global_config.vocab_size
    else:
        # Fallback: extract path if model_path is a model handler object
        if hasattr(model_path, 'model_path'):
            actual_path = model_path.model_path
        else:
            actual_path = model_path

        # Load checkpoint to get configuration
        checkpoint = torch.load(actual_path, map_location='cpu')

        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            block_size = config_dict['block_size']
            vocab_size = config_dict['vocab_size']
        else:
            block_size = 256
            vocab_size = checkpoint.get('vocab_size', 100)

    # Create dummy inputs
    batch_size = 1

    # input_ids: random token indices
    dummy_input_ids = torch.randint(
        0, vocab_size,
        (batch_size, block_size),
        dtype=torch.long
    )

    # sigma: random noise level (typical range for diffusion models)
    dummy_sigma = torch.randn(batch_size, 1, dtype=torch.float32)

    print(f"[Olive] Generated dummy inputs:")
    print(f"  - input_ids: {dummy_input_ids.shape} ({dummy_input_ids.dtype})")
    print(f"  - sigma: {dummy_sigma.shape} ({dummy_sigma.dtype})")

    return (dummy_input_ids, dummy_sigma)


if __name__ == "__main__":
    """Test the loader functions"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/confession.pt',
                       help='Path to model checkpoint')
    args = parser.parse_args()

    print("="*60)
    print("Testing Olive Model Loader Functions")
    print("="*60)

    # Test model loader
    print("\n1. Testing _model_loader:")
    model = _model_loader(args.model)
    print(f"   Model type: {type(model)}")

    # Test io_config
    print("\n2. Testing _io_config:")
    io_config = _io_config(args.model)

    # Test dummy inputs
    print("\n3. Testing _dummy_inputs:")
    dummy_inputs = _dummy_inputs(args.model)

    # Test forward pass
    print("\n4. Testing forward pass with dummy inputs:")
    with torch.no_grad():
        output = model(*dummy_inputs)
    print(f"   Output shape: {output.shape}")
    print(f"   Output dtype: {output.dtype}")

    print("\n" + "="*60)
    print("All tests passed! Ready for Olive optimization.")
    print("="*60)
