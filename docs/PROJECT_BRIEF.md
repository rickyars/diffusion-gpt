# Discrete Diffusion Model: Multi-Dataset Language Training

## Overview
Refactor the Jupyter notebook from `https://github.com/ash80/diffusion-gpt` into a
production-ready local training pipeline that supports training separate character-level
discrete diffusion models on diverse text datasets.

**Source/Inspiration:**
- Repository: https://github.com/ash80/diffusion-gpt
- Paper: "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution"
  (arXiv:2310.16834)
- Original concept: Ashwani Kumar's annotated implementation adapting Karpathy's baby GPT
  into a discrete diffusion model for text generation

## Project Goal
Train 12-15 separate character-level discrete diffusion models, each on a distinct text
corpus. Each trained model becomes an NFT set, where outputs are grouped by training source.
The linguistic fingerprint of each model should reflect its training data (GitHub commits
sound terse; Goodreads reviews sound verbose; arXiv abstracts sound formal, etc.).

## Input Data
User will manually download and place datasets as `.txt` files in `datasets/` directory:
- One document per line (or one paragraph per line)
- Plain text only, no metadata
- Examples: `datasets/github_commits.txt`, `datasets/amazon_reviews.txt`, etc.

## Requirements

### Core Functionality
1. **Data Loading**: Modular dataset loaders that read `.txt` files from `datasets/` directory
2. **Training Pipeline**:
   - Adapt the notebook's discrete diffusion model for character-level text
   - Support training multiple models sequentially or with config file parameterization
   - Save trained model checkpoints with dataset-specific naming
3. **Generation/Inference**: Script to generate samples from trained models
4. **Configuration**: YAML or Python config file to define:
   - Dataset paths and names
   - Model hyperparameters (learning rate, epochs, batch size, vocab size, etc.)
   - Training device (CPU/GPU)
   - Output directories

### Technical Specs
- **Framework**: PyTorch (from original notebook)
- **Model Architecture**: Character-level discrete diffusion model (from ash80/diffusion-gpt)
- **Input**: Text files (one document per line)
- **Output**:
  - Trained model checkpoints (`.pt` files)
  - Generated samples per model (`.txt` files with outputs)
- **Local Training**: CPU-friendly or GPU-accelerated options

### Expected Project Structure
```
diffusion-gpt-local/
├── config.yaml                    # Centralized config for all datasets/runs
├── train.py                       # Main training script
├── generate.py                    # Inference script
├── dataset_loader.py              # Generic dataset loading utilities
├── model.py                       # Discrete diffusion model architecture
├── utils.py                       # Helper functions (tokenization, etc.)
├── datasets/                      # User-provided .txt files
│   ├── github_commits.txt
│   ├── amazon_reviews.txt
│   ├── goodreads_reviews.txt
│   ├── yelp_reviews.txt
│   ├── hacker_news_posts.txt
│   ├── reddit_comments.txt
│   ├── youtube_comments.txt
│   ├── arxiv_abstracts.txt
│   ├── stack_overflow_questions.txt
│   ├── stack_overflow_answers.txt
│   └── [additional datasets as .txt]
├── models/                        # Saved model checkpoints
│   ├── github_commits.pt
│   ├── amazon_reviews.pt
│   ├── goodreads_reviews.pt
│   └── [one .pt per dataset]
├── outputs/                       # Generated samples
│   ├── github_commits_samples.txt
│   ├── amazon_reviews_samples.txt
│   └── [samples per model]
└── README.md                      # Usage instructions
```

### Key Features
1. **Modular Training**: Run `python train.py --dataset github_commits` to train single model,
   or `python train.py --all` to train all datasets in config
2. **Inference**: `python generate.py --model models/github_commits.pt --samples 100`
   generates 100 outputs
3. **Config-Driven**: All hyperparameters in `config.yaml`, no hardcoding
4. **Dataset Agnostic**: Works with any `.txt` file format (one document per line)

### Conversion from Notebook
The original notebook (`Annotated_Discrete_Diffusion_Models.ipynb`) should be:
- Extracted into reusable functions and classes
- Parameterized (move hardcoded values to config)
- Made CLI-driven (argument parsing for `train.py` and `generate.py`)
- Split into logical modules (model architecture, training loop, data loading, generation)

### Deliverables
1. `train.py` — Main training script with CLI args
2. `generate.py` — Inference script
3. `config.yaml` — Configuration template with example datasets
4. `dataset_loader.py` — Generic dataset loading
5. `model.py` — Discrete diffusion model architecture
6. `utils.py` — Shared utilities
7. `README.md` — Setup and usage instructions

## Notes
- User has already selected datasets and knows where to source them (Kaggle links provided separately)
- Training should be deterministic/reproducible (set random seeds)
- Should handle different dataset sizes gracefully
- Model outputs should be directly usable for NFT generation downstream
