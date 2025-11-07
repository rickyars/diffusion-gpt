# Web Demo

Browser-based discrete diffusion text generation using ONNX.js. Deploy to Arweave/IPFS or run locally.

## Quick Start

### 1. Export Model to ONNX

From the project root:
```bash
cd web
python export_to_onnx.py
```

This creates `demo/` with:
- `model.onnx` (~45MB)
- `vocab.json` (<1KB)
- `metadata.json` (<1KB)

### 2. Test Locally

```bash
python test_server.py
```

Open http://localhost:8000/index.html

### 3. Deploy to Arweave/IPFS

Upload these files:
- `demo/model.onnx`
- `demo/vocab.json`
- `demo/metadata.json`

Update URLs in `index.html`, then upload `index.html`.

## Files

- **`index.html`** - Main demo with ONNX.js (auto-play, continuous mode)
- **`diffusion_demo.html`** - Simple JavaScript demo (no model, pure JS simulation)
- **`export_to_onnx.py`** - Converts PyTorch models to ONNX
- **`test_server.py`** - Local development server with CORS headers

## Documentation

See main docs folder:
- `docs/WEB_DEMO_README.md` - Complete documentation
- `docs/QUICK_START_WEB_DEMO.md` - Quick setup guide

## Features

### index.html
- Real trained model inference in browser
- Auto-start mode (begins immediately after loading)
- Continuous mode (loops forever)
- Character-by-character animation
- Time-based seeding (changes every minute)
- Configurable Arweave/IPFS URLs

### diffusion_demo.html
- Pure JavaScript simulation
- No model needed
- Lightweight (~20KB)
- Good for understanding the algorithm

## Requirements

- Modern browser (Chrome, Firefox, Safari, Edge)
- For local testing: Python 3
- For ONNX export: PyTorch, ONNX (see main requirements.txt)

## Usage

### Export Your Own Model

```bash
cd web
python export_to_onnx.py --model ../models/my_model.pt
```

### Test Locally

```bash
python test_server.py
# Open http://localhost:8000/index.html
```

### Deploy

1. Upload `demo/` contents to Arweave/IPFS
2. Note the URLs
3. Edit `index.html` and update the model/vocab/metadata URLs
4. Upload `index.html`
5. Share the URL!

## Notes

- The `demo/` directory is created by `export_to_onnx.py` (not in git)
- Default paths assume you're running from the `web/` directory
- For production, replace `./demo/...` URLs with Arweave/IPFS URLs
