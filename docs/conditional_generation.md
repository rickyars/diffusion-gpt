# Conditional Generation with Prefix and Suffix

## Overview

This implementation adds conditional generation capabilities to the discrete diffusion model, allowing you to prompt the model with specific text at the beginning (prefix) and/or end (suffix) of the generated sequence.

## How It Works

### The Projection Method

Unlike autoregressive models that generate text token-by-token from left to right, discrete diffusion models denoise all tokens simultaneously. To condition generation on specific text:

1. **Encoding**: The prefix and suffix strings are converted to token IDs using the character-level vocabulary
2. **Constraint Mask**: A boolean mask identifies which positions should remain fixed (prefix at start, suffix at end)
3. **Projection Function**: After each denoising step, the constrained positions are reset to their fixed values
4. **Diffusion on Free Tokens**: The model only effectively denoises the unconstrained middle portion

### Implementation Details

#### Key Functions

**`encode(text, stoi)` in utils.py**
- Converts text strings to lists of token indices
- Uses the string-to-index (stoi) vocabulary mapping

**`create_projection_function(prefix_ids, suffix_ids, context_length, device)` in generate.py**
- Creates a closure that constrains specific token positions
- Returns None if no constraints are specified (unconditional generation)
- Returns a function that overwrites constrained positions after each denoising step

**Modified `generate_samples(..., prefix=None, suffix=None)`**
- Accepts optional prefix and suffix strings
- Validates that prefix + suffix don't exceed context length
- Applies projection after each sampling step in the denoising loop

### Comparison with Original Repositories

#### Score-Entropy-Discrete-Diffusion
- Uses `--prefix` and `--suffix` flags for "fill-in-the-middle" generation
- Our implementation follows this approach exactly
- Constrains positions at both ends, fills the middle with diffusion

#### nanoGPT
- Uses `--start` flag for autoregressive prompting
- Generates text left-to-right from the prompt
- Different paradigm (autoregressive vs diffusion)

## Usage Examples

### Basic Prefix Conditioning
```bash
python generate.py \
  --model pretrained_model/model_epoch_25.pth \
  --prefix "Once upon a time" \
  --samples 3
```

The model will generate text that starts with "Once upon a time" and fills in the rest.

### Suffix Conditioning
```bash
python generate.py \
  --model pretrained_model/model_epoch_25.pth \
  --suffix "The End." \
  --samples 3
```

The model will generate text that ends with "The End."

### Fill-in-the-Middle
```bash
python generate.py \
  --model pretrained_model/model_epoch_25.pth \
  --prefix "The answer is" \
  --suffix "and that's final." \
  --samples 5
```

The model will generate coherent text between the prefix and suffix.

### Combined with Other Options
```bash
python generate.py \
  --model pretrained_model/model_epoch_25.pth \
  --prefix "To be or not to be," \
  --steps 256 \
  --samples 10 \
  --output outputs/hamlet_variations.txt \
  --verbose
```

## Technical Notes

### Character-Level Tokenization
- The implementation uses character-level vocabulary (not BPE or word-level)
- Each character in the prefix/suffix must exist in the model's vocabulary
- Unknown characters are silently filtered out during encoding

### Context Length Constraints
- The sum of prefix length + suffix length must be less than the model's context length
- Free tokens = context_length - len(prefix) - len(suffix)
- The script will raise an error if constraints are too long

### Denoising Loop Modification
The projection is applied after each sampling step:
```python
x = sample_categorical(probs)  # Sample next state
if proj_fun is not None:
    x = proj_fun(x)  # Enforce constraints
```

This ensures that even if the model's predictions try to change the constrained tokens, they are immediately reset to the correct values.

## Limitations

1. **Character-Level Only**: Currently works with character-level tokenization only
2. **Fixed Positions**: Prefix must be at the very start, suffix at the very end (no mid-sequence constraints)
3. **No Soft Constraints**: Constraints are hard (tokens are fixed), not soft (weighted preferences)

## Future Enhancements

Potential improvements:
- Support for arbitrary position constraints (not just prefix/suffix)
- Soft conditioning via score adjustment
- Multiple constraint regions
- Gradient-based guidance for semantic conditioning
