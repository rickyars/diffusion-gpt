# Quick Start: Web Demo

Follow these steps to get the web demo running:

## 1. Export Model to ONNX

```bash
# Install dependencies (if not already installed)
pip3 install -r requirements.txt

# Export the model
python3 export_to_onnx.py
```

This creates `web_demo/` with:
- `model.onnx` (~45MB)
- `vocab.json` (<1KB)
- `metadata.json` (<1KB)

## 2. Test Locally

```bash
# Start test server
python3 test_server.py

# Open in browser
# http://localhost:8000/index.html
```

1. Click "Load Model" (uses local files from `web_demo/`)
2. Wait for model to load (~5-10 seconds)
3. Click "Start Generation"
4. Watch text emerge from noise!

## 3. Deploy to Arweave/IPFS

### Upload files:

```bash
# Upload these 3 files and note their URLs:
web_demo/model.onnx
web_demo/vocab.json
web_demo/metadata.json
```

### Update index.html:

Replace the placeholder URLs in the config section (or use the UI):
- Model URL: `https://arweave.net/YOUR_MODEL_TX_ID`
- Vocab URL: `https://arweave.net/YOUR_VOCAB_TX_ID`
- Metadata URL: `https://arweave.net/YOUR_METADATA_TX_ID`

### Upload index.html:

```bash
# Upload index.html to Arweave/IPFS
# Share the URL!
```

## Features

- ✅ Fully self-contained HTML
- ✅ Real PyTorch model running in browser
- ✅ 64-step denoising visualization
- ✅ Character-by-character animation
- ✅ Time-based seed (changes every minute)
- ✅ Works offline after first load
- ✅ No server required

## Troubleshooting

**"Module not found" when exporting:**
```bash
pip3 install torch onnx numpy pyyaml
```

**Model fails to load in browser:**
- Check browser console for errors
- Verify file paths/URLs are correct
- For local testing, use `test_server.py` (not `python -m http.server`)

**Slow inference:**
- Normal on first run (WASM compilation)
- Should be ~1-2 sec/step after warmup
- Consider reducing steps to 32

See WEB_DEMO_README.md for detailed documentation.
