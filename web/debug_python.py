"""
Comprehensive debugging script for comparing Python and ONNX inference.

This script:
1. Tests PyTorch model inference
2. Tests ONNX model inference
3. Compares outputs at each step
4. Identifies discrepancies

USAGE:
    cd web
    python debug_python.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import json
import numpy as np
from model import GPT, GPTConfig

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)

def print_stats(name, data):
    """Print statistics for a tensor or array."""
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    data = np.array(data).flatten()

    print(f"\n{name}:")
    print(f"  Shape: {data.shape}")
    print(f"  Min: {data.min():.6f}")
    print(f"  Max: {data.max():.6f}")
    print(f"  Mean: {data.mean():.6f}")
    print(f"  Std: {data.std():.6f}")
    print(f"  Has NaN: {np.isnan(data).any()}")
    print(f"  Has Inf: {np.isinf(data).any()}")
    print(f"  Sample (first 10): {data[:10]}")

def test_pytorch_inference():
    """Test PyTorch model inference."""
    print_section("TESTING PYTORCH MODEL")

    # Load checkpoint
    checkpoint_path = '../pretrained_model/model_epoch_25.pth'
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Inspect checkpoint structure
    print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Extract config - handle different checkpoint formats
    if 'config' in checkpoint:
        print("\nFound 'config' in checkpoint")
        model_config_dict = checkpoint['config']
        model_config = GPTConfig(**model_config_dict)
    elif 'model_args' in checkpoint:
        print("\nFound 'model_args' in checkpoint")
        model_config_dict = checkpoint['model_args']
        model_config = GPTConfig(**model_config_dict)
    else:
        print("\n⚠️  No config found in checkpoint, trying to infer from state_dict...")
        # Try to infer config from model structure
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))

        # Extract dimensions from the state dict
        if 'transformer.wte.weight' in state_dict:
            vocab_size_inferred = state_dict['transformer.wte.weight'].shape[0]
            n_embd = state_dict['transformer.wte.weight'].shape[1]
        else:
            vocab_size_inferred = 65  # Default Shakespeare vocab
            n_embd = 384  # Default from config

        # Count layers by looking for layer-specific parameters
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

        # Try to infer n_head from attention weights
        if f'transformer.h.0.attn.c_attn.weight' in state_dict:
            # c_attn.weight has shape [n_embd, 3*n_embd] (for Q, K, V)
            n_head = 6  # Default, hard to infer reliably
        else:
            n_head = 6

        print(f"  Inferred: vocab_size={vocab_size_inferred}, n_embd={n_embd}, n_layer={n_layer}, block_size={block_size}")

        model_config = GPTConfig(
            block_size=block_size,
            vocab_size=vocab_size_inferred,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            cond_dim=128,
            dropout=0.0,
            bias=False
        )
        model_config_dict = {
            'block_size': block_size,
            'vocab_size': vocab_size_inferred,
            'n_layer': n_layer,
            'n_head': n_head,
            'n_embd': n_embd,
            'cond_dim': 128,
            'dropout': 0.0,
            'bias': False
        }

    vocab_size = checkpoint.get('vocab_size', model_config.vocab_size)

    print(f"\n  Vocab size: {vocab_size}")
    print(f"  Block size: {model_config.block_size}")
    print(f"  Model config: {model_config_dict}")

    # Initialize model
    print("\nInitializing model...")
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
    print(f"  Parameters: {model.get_num_params() / 1e6:.2f}M")

    # Get vocabulary
    if 'itos' in checkpoint and 'stoi' in checkpoint:
        itos = checkpoint['itos']
        stoi = checkpoint['stoi']
        print(f"\nVocabulary loaded from checkpoint")
    else:
        print("\nWarning: No vocab in checkpoint, using default")
        vocab_str = "\n !\"&'(),-.0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        itos = {i: c for i, c in enumerate(vocab_str)}
        stoi = {c: i for i, c in enumerate(vocab_str)}

    print(f"\nVocabulary sample (first 10):")
    for i in range(min(10, len(itos))):
        char = itos[i]
        display = repr(char) if char in ['\n', ' '] else char
        print(f"  {i}: {display}")

    # Create test input (use fixed seed for reproducibility)
    print("\nCreating test input...")
    torch.manual_seed(42)
    test_input = torch.randint(0, vocab_size, (1, model_config.block_size))
    test_sigma = torch.tensor([10.0])

    print(f"  Input shape: {test_input.shape}")
    print(f"  Input dtype: {test_input.dtype}")
    print(f"  Input sample (first 20): {test_input[0, :20].tolist()}")
    print(f"  Sigma: {test_sigma.item()}")

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        output = model(test_input, test_sigma)

    print_stats("Output logits", output)

    # Get first position in detail
    first_pos_logits = output[0, 0, :].cpu().numpy()
    print_stats("First position logits", first_pos_logits)

    # Convert to scores
    first_pos_scores = np.exp(first_pos_logits)
    print_stats("First position scores (exp)", first_pos_scores)

    # Save test data
    test_data = {
        'input_ids': test_input[0].tolist(),
        'sigma': test_sigma.item(),
        'output_logits': output[0].tolist(),
        'vocab_size': vocab_size,
        'block_size': model_config.block_size,
        'first_pos_logits': first_pos_logits.tolist(),
        'first_pos_scores': first_pos_scores.tolist()
    }

    with open('debug_pytorch_output.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    print("\n✅ Saved PyTorch test data to debug_pytorch_output.json")

    return model, itos, stoi, vocab_size, model_config


def test_onnx_inference():
    """Test ONNX model inference."""
    print_section("TESTING ONNX MODEL")

    try:
        import onnxruntime as ort
    except ImportError:
        print("❌ Error: onnxruntime not installed")
        print("Install with: pip install onnxruntime")
        return None

    onnx_path = './demo/model.onnx'
    if not os.path.exists(onnx_path):
        print(f"❌ Error: ONNX model not found at {onnx_path}")
        print("Run: python export_to_onnx.py")
        return None

    print(f"\nLoading ONNX model: {onnx_path}")
    session = ort.InferenceSession(onnx_path)

    print(f"  Providers: {session.get_providers()}")
    print(f"  Inputs: {[i.name for i in session.get_inputs()]}")
    print(f"  Outputs: {[o.name for o in session.get_outputs()]}")

    # Create same test input as PyTorch
    np.random.seed(42)
    vocab_size = 65  # Should match model
    block_size = 256
    test_input = np.random.randint(0, vocab_size, (1, block_size), dtype=np.int64)
    test_sigma = np.array([10.0], dtype=np.float32)

    print(f"\nTest input shape: {test_input.shape}")
    print(f"Test input dtype: {test_input.dtype}")
    print(f"Test input sample (first 20): {test_input[0, :20].tolist()}")
    print(f"Sigma: {test_sigma[0]}")

    # Run inference
    print("\nRunning ONNX inference...")
    results = session.run(None, {
        'input_ids': test_input,
        'sigma': test_sigma
    })

    output = results[0]
    print_stats("ONNX output logits", output)

    # Get first position
    first_pos_logits = output[0, 0, :]
    print_stats("ONNX first position logits", first_pos_logits)

    # Convert to scores
    first_pos_scores = np.exp(first_pos_logits)
    print_stats("ONNX first position scores (exp)", first_pos_scores)

    # Save ONNX data
    onnx_data = {
        'input_ids': test_input[0].tolist(),
        'sigma': float(test_sigma[0]),
        'output_logits': output[0].tolist(),
        'first_pos_logits': first_pos_logits.tolist(),
        'first_pos_scores': first_pos_scores.tolist()
    }

    with open('debug_onnx_output.json', 'w') as f:
        json.dump(onnx_data, f, indent=2)
    print("\n✅ Saved ONNX test data to debug_onnx_output.json")

    return session


def test_denoising_step(model, itos, stoi, vocab_size, model_config):
    """Test a complete denoising step."""
    print_section("TESTING ONE DENOISING STEP")

    from utils import geometricNoise, staggered_score, transition, sample_categorical

    # Initialize random sequence
    torch.manual_seed(42)
    x = torch.randint(0, vocab_size, (1, model_config.block_size))
    print(f"\nInitial sequence shape: {x.shape}")
    print(f"Initial text (first 100 chars):")
    print(''.join([itos[int(i)] for i in x[0][:100]]))

    # Denoising parameters
    steps = 64
    eps = 1e-5
    current_step = 0
    sigma_min = 0.0001
    sigma_max = 20.0

    # Calculate timestep
    t = 1 - (current_step / steps) * (1 - eps)
    t_next = 1 - ((current_step + 1) / steps) * (1 - eps)

    # Geometric noise
    sigma_min_tensor = torch.tensor(sigma_min)
    sigma_max_tensor = torch.tensor(sigma_max)
    curr_sigma_bar = sigma_min_tensor ** (1 - t) * sigma_max_tensor ** t
    next_sigma_bar = sigma_min_tensor ** (1 - t_next) * sigma_max_tensor ** t_next
    delta_sigma = curr_sigma_bar - next_sigma_bar

    print(f"\nStep {current_step}:")
    print(f"  t = {t:.6f}")
    print(f"  t_next = {t_next:.6f}")
    print(f"  curr_sigma_bar = {curr_sigma_bar:.6f}")
    print(f"  next_sigma_bar = {next_sigma_bar:.6f}")
    print(f"  delta_sigma = {delta_sigma:.6f}")

    # Run model
    print("\nRunning model...")
    with torch.no_grad():
        log_score = model(x, curr_sigma_bar.unsqueeze(0))
        score = torch.exp(log_score)

    print_stats("Log scores", log_score)
    print_stats("Scores (exp)", score)

    # First position in detail
    pos = 0
    current_idx = int(x[0, pos])
    current_char = itos[current_idx]
    print(f"\nPosition {pos}:")
    print(f"  Current token: {current_idx}")
    print(f"  Current char: {repr(current_char)}")

    score_pos = score[0, pos, :].cpu()
    print_stats(f"Position {pos} scores", score_pos)

    # Apply staggered score
    stag_score = staggered_score(score, delta_sigma)
    stag_score_pos = stag_score[0, pos, :].cpu()
    print_stats(f"Position {pos} staggered scores", stag_score_pos)

    # Get transition
    trans = transition(x, delta_sigma, vocab_size)
    trans_pos = trans[0, pos, :].cpu()
    print_stats(f"Position {pos} transition", trans_pos)
    print(f"  Transition sum: {trans_pos.sum():.6f}")
    print(f"  Transition for current idx ({current_idx}): {trans_pos[current_idx]:.6f}")

    # Compute probs
    probs = stag_score * trans
    probs_pos = probs[0, pos, :].cpu()
    print_stats(f"Position {pos} final probs", probs_pos)
    print(f"  Probs sum: {probs_pos.sum():.6f}")

    # Sample
    x_new = sample_categorical(probs)
    new_idx = int(x_new[0, pos])
    new_char = itos[new_idx]
    print(f"\n  Sampled token: {new_idx}")
    print(f"  Sampled char: {repr(new_char)}")
    print(f"  Changed: {new_idx != current_idx}")

    # Full sequence
    num_changed = (x != x_new).sum().item()
    total_tokens = x.numel()
    change_percent = 100 * num_changed / total_tokens

    print(f"\nFull sequence:")
    print(f"  Tokens changed: {num_changed}/{total_tokens} ({change_percent:.1f}%)")
    print(f"  New text (first 100 chars):")
    print(''.join([itos[int(i)] for i in x_new[0][:100]]))

    # Save detailed step data
    step_data = {
        'step': current_step,
        't': float(t),
        't_next': float(t_next),
        'curr_sigma': float(curr_sigma_bar),
        'next_sigma': float(next_sigma_bar),
        'delta_sigma': float(delta_sigma),
        'position_0': {
            'current_idx': current_idx,
            'current_char': current_char,
            'scores': score_pos.numpy().tolist(),
            'staggered_scores': stag_score_pos.numpy().tolist(),
            'transition': trans_pos.numpy().tolist(),
            'final_probs': probs_pos.numpy().tolist(),
            'sampled_idx': new_idx,
            'sampled_char': new_char
        },
        'tokens_changed': num_changed,
        'total_tokens': total_tokens,
        'change_percent': float(change_percent)
    }

    with open('debug_step_output.json', 'w') as f:
        json.dump(step_data, f, indent=2)
    print("\n✅ Saved denoising step data to debug_step_output.json")


def compare_outputs():
    """Compare PyTorch and ONNX outputs."""
    print_section("COMPARING PYTORCH VS ONNX")

    if not os.path.exists('debug_pytorch_output.json'):
        print("❌ PyTorch output not found. Run test first.")
        return

    if not os.path.exists('debug_onnx_output.json'):
        print("❌ ONNX output not found. Run test first.")
        return

    with open('debug_pytorch_output.json', 'r') as f:
        pytorch_data = json.load(f)

    with open('debug_onnx_output.json', 'r') as f:
        onnx_data = json.load(f)

    # Compare inputs
    print("\nComparing inputs:")
    pytorch_input = np.array(pytorch_data['input_ids'])
    onnx_input = np.array(onnx_data['input_ids'])
    print(f"  PyTorch input shape: {pytorch_input.shape}")
    print(f"  ONNX input shape: {onnx_input.shape}")
    print(f"  Inputs match: {np.array_equal(pytorch_input, onnx_input)}")
    print(f"  PyTorch sigma: {pytorch_data['sigma']}")
    print(f"  ONNX sigma: {onnx_data['sigma']}")

    # Compare outputs
    print("\nComparing outputs (first position):")
    pytorch_logits = np.array(pytorch_data['first_pos_logits'])
    onnx_logits = np.array(onnx_data['first_pos_logits'])

    diff = np.abs(pytorch_logits - onnx_logits)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"  PyTorch logits range: [{pytorch_logits.min():.4f}, {pytorch_logits.max():.4f}]")
    print(f"  ONNX logits range: [{onnx_logits.min():.4f}, {onnx_logits.max():.4f}]")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")
    print(f"  Outputs match (tol=1e-5): {np.allclose(pytorch_logits, onnx_logits, atol=1e-5)}")

    if max_diff > 1e-3:
        print("\n⚠️ WARNING: Large differences detected!")
        print("  Top 10 differences:")
        top_diff_idx = np.argsort(diff)[-10:][::-1]
        for idx in top_diff_idx:
            print(f"    Index {idx}: PyTorch={pytorch_logits[idx]:.4f}, ONNX={onnx_logits[idx]:.4f}, Diff={diff[idx]:.6f}")
    else:
        print("\n✅ PyTorch and ONNX outputs match closely!")


def main():
    """Run all tests."""
    print_section("DISCRETE DIFFUSION DEBUG SUITE")
    print("This script tests PyTorch and ONNX inference and identifies issues.")

    # Test PyTorch
    result = test_pytorch_inference()
    if result is None:
        print("\n❌ PyTorch test failed")
        return
    model, itos, stoi, vocab_size, model_config = result

    # Test ONNX
    test_onnx_inference()

    # Compare
    compare_outputs()

    # Test denoising step
    test_denoising_step(model, itos, stoi, vocab_size, model_config)

    print_section("DEBUGGING COMPLETE")
    print("\nGenerated files:")
    print("  - debug_pytorch_output.json")
    print("  - debug_onnx_output.json")
    print("  - debug_step_output.json")
    print("\nNext steps:")
    print("  1. Open web/debug.html in browser")
    print("  2. Load the model")
    print("  3. Click 'Test Single Inference'")
    print("  4. Compare the outputs with the JSON files")
    print("  5. Look for differences in logits, scores, or probabilities")


if __name__ == '__main__':
    main()
