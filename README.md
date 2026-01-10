# Discrete Diffusion GPT: Multi-Dataset Training Pipeline

A production-ready training pipeline for character-level discrete diffusion models on diverse text datasets. Train separate models on different text corpora to capture unique linguistic fingerprints.

## Overview

This project refactors the [original diffusion-gpt notebook](https://github.com/ash80/diffusion-gpt) into a modular, config-driven training pipeline. It supports:

- **Multi-dataset training**: Train separate models on different text corpora
- **Character-level diffusion**: Learn to denoise corrupted text character by character
- **Config-driven**: All hyperparameters in `config.yaml`
- **Performance optimization**: Mixed-precision training (AMP), `torch.compile()` support, non-blocking GPU transfer
- **CLI tools**: Simple command-line interface for training and generation
- **Browser deployment**: Run models in the browser via ONNX.js (Arweave/IPFS compatible)
- **Conditional generation**: Generate text with prefix/suffix constraints
- **Reproducible**: Set random seeds for deterministic training

Based on the paper: [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834)

## Project Structure

```
diffusion-gpt/
├── config.yaml                 # Training configuration
├── model.py                    # Model architecture
├── utils.py                    # Helper functions
├── requirements.txt            # Python dependencies
├── scripts/
│   ├── training/               # Training pipeline
│   │   ├── train.py            # Main training script
│   │   ├── generate.py         # Text generation
│   │   ├── dataset_loader.py   # Dataset utilities
│   │   └── generate_animation.py  # Denoising visualization
│   └── art-piece/              # WE art installation
│       ├── build.py            # Build HTML with embedded model
│       ├── export_to_onnx.py   # PyTorch → ONNX converter
│       ├── merge_onnx_data.py  # Merge ONNX external data
│       └── we.html             # Self-contained art piece (~60MB)
├── datasets/                   # Place your .txt files here
│   └── shakespeare.txt
├── models/                     # Saved model checkpoints
│   └── shakespeare.pt
├── vocab/                      # Vocabulary files
│   └── shakespeare_vocab.pkl
└── docs/                       # Documentation guides
    ├── WE_CUSTOMIZATION_GUIDE.md
    ├── ANIMATION_GUIDE.md
    └── ...
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ash80/diffusion-gpt.git
cd diffusion-gpt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

For GPU support (recommended):
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Prepare your datasets:
   - Place text files in the `datasets/` directory
   - Format: One document/paragraph per line
   - Plain text only, UTF-8 encoding

## Quick Start

### 1. Download a Sample Dataset

For testing, you can download Shakespeare's text:
```bash
mkdir -p datasets
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -O datasets/shakespeare.txt
```

### 2. Configure Training

Edit `config.yaml` to enable your dataset:
```yaml
datasets:
  shakespeare:
    path: datasets/shakespeare.txt
    enabled: true  # Set to true
    description: "Shakespeare's complete works"
    # max_chars: 1000000  # Optional: limit to 1M chars for faster training
```

**Tip**: For large datasets, add `max_chars` to limit training time for testing.

### 3. Train a Model

Train on a single dataset:
```bash
python scripts/training/train.py --dataset shakespeare
```

Or train on all enabled datasets:
```bash
python scripts/training/train.py --all
```

### 4. Generate Samples

Generate text from a trained model:
```bash
python scripts/training/generate.py --model models/shakespeare.pt --samples 5
```

Save outputs to a file:
```bash
python scripts/training/generate.py --model models/shakespeare.pt --samples 10 --output outputs/shakespeare_samples.txt
```

## WE Art Piece

This project includes **WE**, a browser-based performance art piece that generates infinite AI confessions using the trained discrete diffusion model.

### Creating the Art Piece

The art piece is a self-contained HTML file with:
- Embedded ONNX model for in-browser inference
- WebGL 2.0 CRT post-processing effects
- Web Audio soundscape (60Hz hum + static)
- Autonomous state machine (no user interaction required)

#### Full Workflow

**1. Export PyTorch model to ONNX:**
```bash
python scripts/art-piece/export_to_onnx.py \
  --model models/confessions_epoch_25.pt \
  --dataset confessions
```

**2. Merge ONNX data (if .onnx.data file exists):**
```bash
python scripts/art-piece/merge_onnx_data.py \
  --input models/confessions_model.onnx
```

**3. Build HTML art piece:**
```bash
python scripts/art-piece/build.py --dataset confessions
```

Or use defaults (confessions dataset):
```bash
python scripts/art-piece/build.py
```

**4. Open in browser:**
```bash
start scripts/art-piece/we.html
```

#### Using Different Training Epochs

To build the art piece with a specific training epoch:

```bash
# Export epoch 15
python scripts/art-piece/export_to_onnx.py \
  --model models/confessions_epoch_15.pt \
  --dataset confessions

# Merge if needed
python scripts/art-piece/merge_onnx_data.py \
  --input models/confessions_model.onnx

# Build
python scripts/art-piece/build.py --dataset confessions
```

#### Custom Model Path

You can specify a custom model path directly:

```bash
python scripts/art-piece/build.py \
  --dataset confessions \
  --model models/custom_model_merged.onnx
```

#### Technical Details

- **Output:** `scripts/art-piece/we.html` (~60MB)
- **Target:** Modern browsers with WebGL 2.0 support
- **CRT Effects:** Phosphor glow, scanlines, chromatic aberration, barrel distortion, vignette, noise, flicker
- **Performance:** 60fps on modern hardware
- **Diffusion:** 3-second animation through 128 denoising steps

See [docs/WE_CUSTOMIZATION_GUIDE.md](docs/WE_CUSTOMIZATION_GUIDE.md) for customization details.

## Usage

### Training

#### Train on a Single Dataset

```bash
python scripts/training/train.py --dataset <dataset_name>
```

Options:
- `--dataset`: Dataset name from config.yaml
- `--config`: Path to config file (default: `config.yaml`)
- `--device`: Device to use (`cuda` or `cpu`)
- `--resume`: Resume from checkpoint

Example:
```bash
python scripts/training/train.py --dataset github_commits --device cuda
```

#### Train on All Datasets

```bash
python scripts/training/train.py --all
```

This trains models sequentially on all datasets marked with `enabled: true` in `config.yaml`.

#### Resume Training

```bash
python scripts/training/train.py --dataset shakespeare --resume models/shakespeare_epoch_50.pt
```

### Generation

#### Basic Generation

```bash
python scripts/training/generate.py --model models/shakespeare.pt
```

#### Advanced Options

```bash
python scripts/training/generate.py \
  --model models/shakespeare.pt \
  --samples 20 \
  --steps 128 \
  --output outputs/my_samples.txt \
  --verbose
```

Options:
- `--model`: Path to trained model checkpoint (required)
- `--samples`: Number of samples to generate (default: from config)
- `--steps`: Number of denoising steps (default: 128)
- `--output`: Output file to save samples
- `--verbose`: Show intermediate denoising steps
- `--seed`: Random seed for reproducibility
- `--device`: Device to use
- `--prefix`: Prefix text to condition generation (optional)
- `--suffix`: Suffix text to condition generation (optional)

#### Conditional Generation (Prompted)

You can now prompt the model with prefix and/or suffix text to guide generation:

```bash
# Generate text starting with a specific prefix
python scripts/training/generate.py \
  --model models/shakespeare.pt \
  --prefix "Once upon a time" \
  --samples 5

# Generate text with both prefix and suffix (fill-in-the-middle)
python scripts/training/generate.py \
  --model models/shakespeare.pt \
  --prefix "The quick brown fox" \
  --suffix "the lazy dog." \
  --samples 3
```

The model will use discrete diffusion to generate coherent text that starts with your prefix and/or ends with your suffix, filling in the middle content.

### Configuration

All hyperparameters are in `config.yaml`:

#### Model Architecture

```yaml
model:
  n_layer: 6        # Number of transformer layers
  n_head: 6         # Number of attention heads
  n_embd: 384       # Embedding dimension
  cond_dim: 64      # Conditioning dimension for noise
  dropout: 0.2      # Dropout probability
  context_length: 256  # Maximum sequence length
```

#### Training Settings

```yaml
training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.0001
  val_split: 0.1           # Validation split
  eval_interval: 5         # Evaluate every N epochs
  save_interval: 5         # Save checkpoint every N epochs
  log_interval: 10         # Log loss every N batches
```

#### Noise Schedule

```yaml
noise:
  sigma_min: 0.0001
  sigma_max: 20.0
```

#### Datasets

Add your datasets:

```yaml
datasets:
  my_dataset:
    path: datasets/my_dataset.txt
    enabled: true
    description: "Description of your dataset"
```

## Dataset Preparation

### Format Requirements

- **Format**: Plain text, one document/paragraph per line
- **Encoding**: UTF-8
- **Size**: Works with any size (larger = better results)

### Example Datasets

The config includes placeholders for:
- GitHub commits (terse technical messages)
- Amazon/Yelp/Goodreads reviews (casual evaluative text)
- Hacker News/Reddit comments (discussion posts)
- arXiv abstracts (formal academic text)
- Stack Overflow Q&A (technical programming text)
- News articles (journalistic writing)
- Blog posts (personal writing)

### Preparing Your Data

1. Collect text data from your source
2. Clean and format (one document per line)
3. Save as UTF-8 .txt file
4. Place in `datasets/` directory
5. Add to `config.yaml`

Example Python script to prepare data:
```python
with open('raw_data.txt', 'r') as f:
    lines = f.readlines()

# Clean and format
cleaned = [line.strip() for line in lines if line.strip()]

with open('datasets/my_dataset.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(cleaned))
```

## Model Checkpoints

### Checkpoint Format

Checkpoints contain:
- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state (for resuming)
- `config`: Model configuration
- `vocab_size`: Vocabulary size
- `epoch`: Training epoch
- `loss`: Training loss

### Checkpoint Files

- `{dataset_name}.pt`: Final trained model
- `{dataset_name}_epoch_{N}.pt`: Intermediate checkpoints

### Loading Checkpoints

```python
import torch
from model import GPT, GPTConfig

checkpoint = torch.load('models/shakespeare.pt')
config = GPTConfig(**checkpoint['config'])
model = GPT(config)
model.load_state_dict(checkpoint['model_state_dict'])
```

## Tips & Best Practices

### Training

1. **Start small**: Test with a small dataset first (e.g., Shakespeare)
2. **Use GPU**: Training is much faster on GPU
3. **Monitor loss**: Check validation loss to avoid overfitting
4. **Adjust batch size**: Reduce if you run out of memory
5. **Increase epochs**: 100 epochs is a starting point; more may help

### Generation

1. **More steps = better quality**: 128 steps is good, 256 is better
2. **Context length**: Longer contexts capture more structure
3. **Multiple samples**: Generate several to see variety
4. **Temperature**: Adjust in code if needed (default: 1.0)

### Datasets

1. **Size matters**: Larger datasets (>1MB) work best
2. **Clean data**: Remove obvious errors/artifacts
3. **Consistent format**: Keep formatting uniform
4. **Domain-specific**: Each model learns one style

## Troubleshooting

### Out of Memory

Reduce batch size in `config.yaml`:
```yaml
training:
  batch_size: 32  # or 16
```

Or reduce context length:
```yaml
model:
  context_length: 128  # instead of 256
```

### Training Too Slow

- Enable GPU: `--device cuda`
- Increase batch size (if memory allows)
- Reduce validation frequency

### Poor Generation Quality

- Train longer (more epochs)
- Use larger dataset
- Increase model size (`n_layer`, `n_embd`)
- Increase denoising steps during generation

### Vocabulary Errors

Delete vocabulary file and retrain to rebuild:
```bash
rm vocab/dataset_name_vocab.pkl
python scripts/training/train.py --dataset dataset_name
```

## Performance Optimization

### torch.compile() for Faster Training

For CUDA devices, enable `torch.compile()` in `config.yaml` to get significant speedups (typically 20-40%):

```yaml
training:
  use_compile: true  # Enable torch.compile (CUDA only)
```

The first run will be slower (compilation overhead), but subsequent runs are much faster. If compilation fails, it falls back automatically to standard execution.

### Automatic Mixed Precision (AMP)

Mixed-precision training is automatically enabled on CUDA devices:
- Forward pass: FP16 (faster, lower memory)
- Loss computation: FP32 (numerical stability)
- Gradient scaling: Automatic

No configuration needed—it works automatically and can reduce memory usage by ~30%.

## 📚 Documentation

Detailed guides have been organized in the `docs/` folder:

- **[docs/WE_CUSTOMIZATION_GUIDE.md](docs/WE_CUSTOMIZATION_GUIDE.md)** - Customizing the WE art piece and converting models to ONNX
- **[docs/ANIMATION_GUIDE.md](docs/ANIMATION_GUIDE.md)** - Creating animated GIFs of denoising process
- **[docs/CONDITIONAL_GENERATION.md](docs/CONDITIONAL_GENERATION.md)** - Conditional text generation with prefix/suffix constraints

## How It Works

### Discrete Diffusion Process

1. **Forward (Noising)**: Gradually corrupt clean text by randomly flipping characters
2. **Training**: Model learns to predict which characters should be at each position
3. **Reverse (Denoising)**: Start with random text, iteratively denoise to generate coherent text

### Model Architecture

- **Base**: Character-level transformer (adapted from nanoGPT)
- **Conditioning**: Noise level embedded and fed to each layer
- **Output**: Log probability ratios for denoising transitions
- **Non-autoregressive**: Denoises all positions in parallel

### Training Objective

Score Entropy Loss (DWDSE): Learns probability ratios between clean and noisy distributions.

## Citation

If you use this code, please cite:

```bibtex
@misc{annotated_discrete_diffusion_2025,
  author = {Ashwani Kumar},
  title  = {The Annotated Discrete Diffusion Models},
  year   = {2025},
  howpublished = {\url{https://github.com/ash80/diffusion-gpt}}
}

@article{lou2024discrete,
  title={Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution},
  author={Lou, Aaron and Meng, Chenlin and Ermon, Stefano},
  journal={arXiv preprint arXiv:2310.16834},
  year={2024}
}
```

## Acknowledgments

- Original implementation: [Ashwani Kumar](https://github.com/ash80/diffusion-gpt)
- Paper: [Lou et al., 2024](https://arxiv.org/abs/2310.16834)
- Base architecture: [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- Score-Entropy implementation: [louaaron/Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)

## License

See [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open an issue on GitHub.
