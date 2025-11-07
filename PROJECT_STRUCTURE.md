# Project Structure

Quick reference guide explaining what each file in this project does.

---

## 📁 Core Training & Inference Files

### `train.py` ⭐ **Main training script**
**What it does**: Trains discrete diffusion models on your text datasets.

**Why you need it**: This is how you train models! It:
- Loads text data from `datasets/`
- Trains a discrete diffusion model
- Saves checkpoints to `models/`
- Creates vocabularies in `vocab/`

**Usage**:
```bash
python train.py
```

---

### `generate.py` ⭐ **Text generation script**
**What it does**: Generates new text from a trained model.

**Why you need it**: After training, use this to generate Shakespeare-style text (or whatever you trained on).

**Usage**:
```bash
python generate.py --model models/shakespeare.pt --samples 5
```

---

### `dataset_loader.py` 🔧 **Data loading utility**
**What it does**:
- Reads `.txt` files from `datasets/`
- Builds character vocabularies (maps each character to a number)
- Splits data into train/validation sets
- Creates PyTorch DataLoaders for training

**Why you need it**: `train.py` uses this to load and prepare your text data. It handles all the data preprocessing so your model can train on it.

**You don't run this directly** - it's imported by `train.py`.

---

### `model.py` 🔧 **Model architecture**
**What it does**: Defines the neural network architecture (GPT-based discrete diffusion model).

**Why you need it**: Contains the actual model code. Both `train.py` and `generate.py` import this.

**You don't run this directly** - it's imported by other scripts.

---

### `utils.py` 🔧 **Helper functions**
**What it does**: Utility functions for:
- Noise schedules (geometric noise)
- Text encoding/decoding
- Perturbation functions (adding noise)
- Sampling functions

**Why you need it**: Contains the math and helper functions that make discrete diffusion work.

**You don't run this directly** - it's imported by other scripts.

---

## 🌐 Web Demo Files

### `index.html` ⭐ **Browser-based demo**
**What it does**: Standalone HTML page that runs the model in your browser using ONNX.js.

**Why you need it**: For deploying to Arweave/IPFS or running demos without Python.

**Usage**: Open in browser after exporting model to ONNX.

---

### `export_to_onnx.py` ⭐ **Model export script**
**What it does**: Converts PyTorch `.pt` models to ONNX format for browser use.

**Why you need it**: Required before using `index.html`. Creates:
- `web_demo/model.onnx` (browser-compatible model)
- `web_demo/vocab.json` (character mappings)
- `web_demo/metadata.json` (config)

**Usage**:
```bash
python export_to_onnx.py
```

---

### `test_server.py` 🔧 **Local development server**
**What it does**: Simple HTTP server with proper CORS headers for testing the web demo locally.

**Why you need it**: For testing `index.html` before deploying to Arweave.

**Usage**:
```bash
python test_server.py
# Then open http://localhost:8000/index.html
```

---

## 🎨 Visualization

### `generate_animation.py` ⭐ **GIF animation creator**
**What it does**: Creates animated GIFs showing the denoising process step-by-step.

**Why you need it**: Makes cool visualizations of text emerging from noise (like the demo GIFs).

**Usage**:
```bash
python generate_animation.py --model models/shakespeare.pt
```

---

## 🔍 Utility Scripts

### `inspect_model.py` 🔧 **Model inspector**
**What it does**: Prints info about a trained model checkpoint (config, vocab size, etc.).

**Why you need it**: Quick way to see what's in a `.pt` or `.pth` file.

**Usage**:
```bash
python inspect_model.py
```

---

### `csv_to_dataset.py` 🔧 **CSV converter**
**What it does**: Converts CSV files to `.txt` format for training.

**Why you need it**: If you have data in CSV format, this extracts text columns.

**Usage**:
```bash
python csv_to_dataset.py --input data.csv --column text
```

---

## ⚙️ Configuration Files

### `config.yaml` ⭐ **Main configuration**
**What it does**: Contains all training settings:
- Model architecture (layers, heads, dimensions)
- Training hyperparameters (batch size, learning rate, epochs)
- Dataset paths
- Noise schedule settings

**Why you need it**: Edit this to change training settings or add new datasets.

---

### `requirements.txt` 📦 **Python dependencies**
**What it does**: Lists required Python packages.

**Why you need it**: Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## 📚 Documentation (`docs/` folder)

All the detailed guides have been moved here to declutter the main directory:

- **`ANIMATION_GUIDE.md`**: How to create animated GIFs
- **`CPU_INFERENCE_GUIDE.md`**: Guide for running on CPU
- **`PERFORMANCE.md`**: Performance tips and benchmarks
- **`PROJECT_BRIEF.md`**: Original project brief
- **`QUICK_START_WEB_DEMO.md`**: Quick web demo setup
- **`WEB_DEMO_README.md`**: Comprehensive web demo docs
- **`ORIGINAL_NOTEBOOK_README.md`**: Info about the original Jupyter notebook

---

## 📓 Notebooks

### `The_Annotated_Discrete_Diffusion_Models.ipynb` 📖
**What it does**: Original all-in-one Jupyter notebook with theory and implementation.

**Why you have it**: Educational reference showing how everything works in one place.

---

## 🗂️ Data Directories

### `datasets/` - Training data
Put your `.txt` files here (one document per line).

### `models/` - Trained models
`.pt` files saved by `train.py` go here.

### `vocab/` - Character vocabularies
`.pkl` files with character-to-index mappings.

### `outputs/` - Generated text & animations
Output from `generate.py` and `generate_animation.py`.

### `web_demo/` - ONNX exports
Created by `export_to_onnx.py` for browser inference.

### `pretrained_model/` - Pre-trained weights
Contains `model_epoch_25.pth` (trained on Shakespeare).

---

## 🚀 Quick Start Summary

**To train a model:**
1. Put text in `datasets/your_text.txt`
2. Add entry to `config.yaml`
3. Run: `python train.py`

**To generate text:**
```bash
python generate.py --model models/your_text.pt
```

**To create animations:**
```bash
python generate_animation.py --model models/your_text.pt
```

**To deploy to web:**
```bash
python export_to_onnx.py
python test_server.py  # test locally
# Then upload to Arweave/IPFS
```

---

## 🧹 Files You Can Ignore

- `.git/` - Git version control (don't touch)
- `.gitignore` - Files to ignore in git
- `LICENSE` - Software license
- `*.pyc`, `__pycache__/` - Python cache files
- `.DS_Store` - macOS metadata

---

## ❓ Still Confused?

**"I want to train on my own text"**
→ Use `train.py` + `config.yaml`

**"I want to generate text"**
→ Use `generate.py`

**"I want to deploy to Arweave"**
→ Use `export_to_onnx.py` + `index.html`

**"I want cool GIF animations"**
→ Use `generate_animation.py`

**"What does [file] do?"**
→ Check this guide!
