# Limiting Dataset Size

If you have very large datasets that would take too long to train on, you can limit how many characters are loaded.

## Quick Example

In `config.yaml`:

```yaml
datasets:
  amazon_reviews:
    path: datasets/amazon_reviews.txt
    enabled: true
    description: "Customer product reviews"
    max_chars: 10000000  # Only use first 10M characters (~10MB)
```

## Why Limit Dataset Size?

**Training time**: Smaller datasets train faster
- 1M characters: ~30 min - 1 hour
- 10M characters: ~3-5 hours
- 100M characters: ~24+ hours

**Memory**: Large datasets use more RAM
- 10M characters: ~200MB RAM
- 100M characters: ~2GB RAM

**Quick experiments**: Test hyperparameters on a subset before full training

## Usage

Add `max_chars` to any dataset in `config.yaml`:

```yaml
datasets:
  my_dataset:
    path: datasets/my_large_file.txt
    enabled: true
    max_chars: 5000000  # Limit to 5M characters
```

The script will:
1. Load only the first N characters from the file
2. Show you how much was truncated
3. Train on the limited subset

## Recommended Limits

### For Quick Testing (minutes)
```yaml
max_chars: 100000  # 100K chars ~= 100KB
```

### For Small Models (< 1 hour)
```yaml
max_chars: 1000000  # 1M chars ~= 1MB
```

### For Medium Datasets (few hours)
```yaml
max_chars: 10000000  # 10M chars ~= 10MB
```

### For Large Datasets (many hours)
```yaml
max_chars: 50000000  # 50M chars ~= 50MB
```

### No Limit (use entire file)
```yaml
# Simply don't include max_chars, or comment it out
# max_chars: 10000000
```

## Example Output

When you set a limit, you'll see:

```
Loading training data...
Limiting dataset to 10,000,000 characters (original: 125,348,912)
Loaded 10,000,000 characters
train split has 9,000,000 characters
```

## Tips

### Start Small, Then Scale Up

```yaml
# First run: test quickly
max_chars: 100000

# Second run: verify it works
max_chars: 1000000

# Final run: full training
# max_chars: 50000000  # or remove this line
```

### Different Limits for Different Datasets

```yaml
datasets:
  # Small dataset - use all of it
  shakespeare:
    path: datasets/shakespeare.txt
    enabled: true
    # No max_chars - it's already small

  # Large dataset - limit it
  wikipedia:
    path: datasets/wikipedia_en.txt
    enabled: true
    max_chars: 20000000  # Only 20M chars

  # Testing dataset - very small
  test:
    path: datasets/test_data.txt
    enabled: false
    max_chars: 10000  # Just for quick tests
```

### Estimating Characters

Rough estimate:
- **1,000 characters** ≈ 1KB ≈ 200 words ≈ 1 paragraph
- **10,000 characters** ≈ 10KB ≈ 2,000 words ≈ 3-4 pages
- **100,000 characters** ≈ 100KB ≈ 20,000 words ≈ 30-40 pages
- **1,000,000 characters** ≈ 1MB ≈ 200,000 words ≈ 300-400 pages
- **10,000,000 characters** ≈ 10MB ≈ 2,000,000 words ≈ 3,000-4,000 pages

### Balancing Quality vs Speed

More data usually = better quality, but:
- **< 100K chars**: Model may not learn much
- **100K - 1M chars**: Decent for testing/prototypes
- **1M - 10M chars**: Good quality results
- **10M+ chars**: Diminishing returns (more data helps less)

For most use cases, **1-10M characters** is the sweet spot.

## Training Time Estimates

On a typical GPU (e.g., RTX 3080):

| Characters | File Size | Training Time (25 epochs) |
|------------|-----------|---------------------------|
| 100K       | ~100KB    | ~5-10 minutes             |
| 1M         | ~1MB      | ~30-60 minutes            |
| 10M        | ~10MB     | ~3-5 hours                |
| 50M        | ~50MB     | ~12-24 hours              |
| 100M+      | ~100MB+   | ~24-48+ hours             |

On CPU: multiply by 10-20x

## Complete Example

```yaml
datasets:
  # Production: full dataset
  production_model:
    path: datasets/all_my_data.txt
    enabled: false  # disable for now
    description: "Full production dataset"
    # No limit - use everything

  # Development: limited for fast iteration
  dev_model:
    path: datasets/all_my_data.txt  # Same file!
    enabled: true
    description: "Development model (limited)"
    max_chars: 5000000  # Only 5M chars for fast training
```

This way you can:
1. Quickly iterate with `dev_model` (limited)
2. Final training with `production_model` (full dataset)
3. Both use the same source file!

## See It In Action

When training starts:

```bash
python train.py --dataset my_dataset
```

Output:
```
Training model on: my_dataset
Dataset path: datasets/my_dataset.txt
Dataset size limit: 10,000,000 characters

Loading training data...
Limiting dataset to 10,000,000 characters (original: 85,234,192)
Loaded 10,000,000 characters
train split has 9,000,000 characters
val split has 1,000,000 characters

Training...
```

Now you can train on huge datasets without waiting days! 🚀
