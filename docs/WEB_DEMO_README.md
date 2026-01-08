# Web Demo - Discrete Diffusion Text Generation

This directory contains a standalone HTML demo that runs discrete diffusion text generation directly in the browser using ONNX.js.

## Files

- **index.html** - Main demo page with ONNX.js inference
- **export_to_onnx.py** - Script to convert PyTorch model to ONNX
- **test_server.py** - Simple HTTP server for local testing
- **inspect_model.py** - Utility to inspect model checkpoint

## Quick Start

### Step 1: Export the Model to ONNX

First, install dependencies:

```bash
pip install torch onnx numpy pyyaml
```

Then export the model:

```bash
python3 export_to_onnx.py
```

This will create a `web_demo/` directory with:
- `model.onnx` - ONNX format model (~45MB)
- `vocab.json` - Character vocabulary
- `metadata.json` - Model configuration

### Step 2: Test Locally

Run the test server:

```bash
python3 test_server.py
```

Then open http://localhost:8000/index.html in your browser.

### Step 3: Deploy to Arweave/IPFS

1. Upload the three files to Arweave or IPFS:
   - `web_demo/model.onnx`
   - `web_demo/vocab.json`
   - `web_demo/metadata.json`

2. Get the permanent URLs for each file (e.g., `https://arweave.net/[transaction-id]`)

3. Update the URLs in `index.html` (or use the UI configuration section)

4. Upload `index.html` to Arweave/IPFS

## How It Works

### Architecture

```
┌──────────────────┐
│   index.html     │
│   (Browser)      │
└────────┬─────────┘
         │
         ├─> Load model.onnx (from Arweave/IPFS)
         ├─> Load vocab.json
         └─> Load metadata.json
         │
         ├─> Initialize ONNX.js runtime
         ├─> Create random noise sequence
         └─> Run denoising loop (64 steps)
             │
             ├─> Step 1: σ = 20.0   (pure noise)
             ├─> Step 2: σ = 18.5
             ├─> ...
             └─> Step 64: σ = 0.0001 (clean text)
```

### Diffusion Process

1. **Initialization**: Start with 256 random characters
2. **Noise Schedule**: Geometric schedule from σ_max=20.0 to σ_min=0.0001
3. **Denoising Steps**: 64 iterative refinement steps
4. **Model Inference**: At each step:
   - Feed noisy sequence + noise level to ONNX model
   - Get log probability scores for each character
   - Apply staggered score correction
   - Sample from transition probabilities
   - Update sequence

### Seeding

The demo uses a time-based seed that changes every minute:
- Seed = floor(current_timestamp_ms / 60000)
- This ensures reproducibility within each minute
- New generations every minute use a fresh seed

## ONNX Export Details

The export script (`export_to_onnx.py`) converts the PyTorch model with:

- **Opset version**: 15 (compatible with ONNX.js 1.16+)
- **Dynamic axes**: Batch size is dynamic (sequence length is fixed)
- **Input tensors**:
  - `input_ids`: [batch_size, seq_length] int64
  - `sigma`: [batch_size] float32
- **Output tensor**:
  - `logits`: [batch_size, seq_length, vocab_size] float32

## Browser Compatibility

Tested on:
- Chrome 120+
- Firefox 120+
- Safari 17+
- Edge 120+

Requires:
- WebAssembly support
- ~100MB RAM for model inference
- Modern JavaScript (ES6+)

## File Sizes

- **index.html**: ~20KB (standalone, includes all JS)
- **model.onnx**: ~45MB (trained model weights)
- **vocab.json**: <1KB (character mappings)
- **metadata.json**: <1KB (model config)

**Total download**: ~45MB (one-time, cached by browser)

## Troubleshooting

### Model fails to load

- Check browser console for errors
- Verify URLs are accessible (no CORS issues)
- For Arweave, wait for transaction confirmation

### ONNX.js errors

- Ensure using a modern browser
- Try clearing browser cache
- Check that model.onnx is valid ONNX format

### Slow inference

- Normal for first run (model compilation)
- Subsequent runs should be faster (~1-2 sec per step)
- Consider reducing steps from 64 to 32 for faster results

### CORS issues on local testing

- Use the provided `test_server.py` instead of `python -m http.server`
- It sets proper CORS headers for ONNX.js

## Advanced Configuration

### Changing generation parameters

Edit in `index.html`:

```javascript
const config = {
    steps: 64,          // Number of denoising steps
    sigmaMin: 0.0001,  // Minimum noise level
    sigmaMax: 20.0,    // Maximum noise level
    delayMs: 150       // Visualization delay (ms)
};
```

### Using different models

1. Export your model with `export_to_onnx.py`
2. Update URLs in the HTML
3. Ensure vocabulary and metadata match

## License

Same as main repository (see LICENSE file)

## Credits

- Model architecture based on [Discrete Diffusion Modeling](https://arxiv.org/abs/2310.16834)
- ONNX.js by Microsoft
- Demo by Ashwani Kumar
