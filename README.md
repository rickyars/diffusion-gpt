# Discrete Diffusion GPT: Multi-Dataset Training Pipeline

A production-ready training pipeline for character-level discrete diffusion models on diverse text datasets. Train separate models on different text corpora to capture unique linguistic fingerprints.

## Overview

This project refactors the [original diffusion-gpt notebook](https://github.com/ash80/diffusion-gpt) into a modular, config-driven training pipeline. It supports:

- **Multi-dataset training**: Train separate models on different text corpora
- **Character-level diffusion**: Learn to denoise corrupted text character by character
- **Config-driven**: All hyperparameters in `config.yaml`
- **CLI tools**: Simple command-line interface for training and generation
- **Browser deployment**: Run models in the browser via ONNX.js (Arweave/IPFS compatible)
- **Reproducible**: Set random seeds for deterministic training

Based on the paper: [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834)

📖 **New to the project?** See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for a guide to what each file does.

## Project Structure

```
diffusion-gpt/
├── config.yaml                 # Training configuration
├── train.py                    # Main training script
├── generate.py                 # Text generation script
├── dataset_loader.py           # Dataset loading utilities
├── model.py                    # Model architecture
├── utils.py                    # Helper functions
├── requirements.txt            # Python dependencies
├── datasets/                   # Place your .txt files here
│   ├── shakespeare.txt
│   ├── github_commits.txt
│   └── ...
├── models/                     # Saved model checkpoints
│   ├── shakespeare.pt
│   └── ...
├── vocab/                      # Vocabulary files
│   ├── shakespeare_vocab.pkl
│   └── ...
└── outputs/                    # Generated samples
    ├── shakespeare_samples.txt
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
```

### 3. Train a Model

Train on a single dataset:
```bash
python train.py --dataset shakespeare
```

Or train on all enabled datasets:
```bash
python train.py --all
```

### 4. Generate Samples

Generate text from a trained model:
```bash
python generate.py --model models/shakespeare.pt --samples 5
```

Save outputs to a file:
```bash
python generate.py --model models/shakespeare.pt --samples 10 --output outputs/shakespeare_samples.txt
```

## Usage

### Training

#### Train on a Single Dataset

```bash
python train.py --dataset <dataset_name>
```

Options:
- `--dataset`: Dataset name from config.yaml
- `--config`: Path to config file (default: `config.yaml`)
- `--device`: Device to use (`cuda` or `cpu`)
- `--resume`: Resume from checkpoint

Example:
```bash
python train.py --dataset github_commits --device cuda
```

#### Train on All Datasets

```bash
python train.py --all
```

This trains models sequentially on all datasets marked with `enabled: true` in `config.yaml`.

#### Resume Training

```bash
python train.py --dataset shakespeare --resume models/shakespeare_epoch_50.pt
```

### Generation

#### Basic Generation

```bash
python generate.py --model models/shakespeare.pt
```

#### Advanced Options

```bash
python generate.py \
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
python train.py --dataset dataset_name
```

## 📚 Documentation

Detailed guides have been organized in the `docs/` folder:

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - 🌟 **START HERE!** Explains what every file does
- **[docs/WEB_DEMO_README.md](docs/WEB_DEMO_README.md)** - Complete web demo documentation
- **[docs/QUICK_START_WEB_DEMO.md](docs/QUICK_START_WEB_DEMO.md)** - Fast setup for browser inference
- **[docs/ANIMATION_GUIDE.md](docs/ANIMATION_GUIDE.md)** - Creating animated GIFs
- **[docs/CPU_INFERENCE_GUIDE.md](docs/CPU_INFERENCE_GUIDE.md)** - Running on CPU
- **[docs/PERFORMANCE.md](docs/PERFORMANCE.md)** - Performance tips and benchmarks
- **[docs/ORIGINAL_NOTEBOOK_README.md](docs/ORIGINAL_NOTEBOOK_README.md)** - About the Jupyter notebook

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
