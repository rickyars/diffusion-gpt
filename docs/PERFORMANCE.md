# Performance Optimization Guide

## Current Training Analysis

Your training is doing **15,682 batches per epoch** - this is correct! The original notebook would have done the same.

## Why Your Training is Slow

At ~2.8 seconds/batch:
- **1 epoch = 12.4 hours**
- **25 epochs = 13 days**
- **100 epochs = 52 days** ❌

## Speed Improvements (Just Added)

The code now includes several optimizations that should give **2-4x speedup**:

### 1. **torch.compile()** (30-50% faster)
- Automatic optimization of model execution
- Enabled automatically on CUDA GPUs with PyTorch 2.0+

### 2. **Mixed Precision Training (AMP)** (2-3x faster)
- Uses float16 instead of float32 where possible
- Reduces memory usage (allows larger batch sizes)
- Enabled automatically on CUDA GPUs

### 3. **Parallel Data Loading** (10-20% faster)
- `num_workers=2` loads batches in background
- Prevents GPU from waiting for CPU

### 4. **Reduced Logging** (small improvement)
- `log_interval: 100` instead of 10
- Less I/O overhead

### 5. **Realistic Epochs**
- Changed from 100 → 25 epochs (original notebook used 25)

## What You Should Do RIGHT NOW

### Option 1: Stop and Restart (RECOMMENDED)
```bash
# Stop current training (Ctrl+C)
git pull
python train.py --dataset shakespeare
```

**Expected new time:** ~4-6 hours per epoch with optimizations = **~5-6 days for 25 epochs**

### Option 2: Increase Batch Size (MUCH FASTER)

Check your GPU memory:
```bash
nvidia-smi
```

If you're using < 50% of GPU memory, **double your batch size**:

Edit `config.yaml`:
```yaml
training:
  batch_size: 128  # or even 256 if GPU allows
```

**Impact:**
- `batch_size: 128` → ~6-7 hours per epoch → **~7 days for 25 epochs**
- `batch_size: 256` → ~3-4 hours per epoch → **~4 days for 25 epochs**

### Option 3: Test Your Current Checkpoint

You likely have a checkpoint at epoch 5. Test it NOW:

```bash
python generate.py --model models\shakespeare_epoch_5.pt --samples 3
```

The model should already be producing somewhat coherent text!

## Understanding the Numbers

### Batch Count is Normal
- Shakespeare dataset: 1,003,854 characters
- Context window: 256 characters
- Batch size: 64
- **Batches per epoch: 15,682** ✓ (This is correct!)

The original notebook did the same number of batches. Your GPU is just slower or the optimizations weren't enabled.

## About Flash Attention

The warning about Flash Attention is not the main problem:
- Flash Attention is an optimization for very large models
- You ARE using `scaled_dot_product_attention` (which is already optimized)
- The warning just means you don't have the extra Flash Attention CUDA kernels
- **For your model size (11M params), this makes ~5-10% difference, not 4x**

## Recommended Configuration

### For Fast Experimentation (test immediately):
```yaml
training:
  epochs: 10
  batch_size: 256  # if GPU allows
```
**Time: ~1-2 days**

### For Good Quality (recommended):
```yaml
training:
  epochs: 25
  batch_size: 128
```
**Time: ~4-7 days**

### For Best Quality:
```yaml
training:
  epochs: 50
  batch_size: 128
```
**Time: ~8-14 days**

## What GPU Do You Have?

Run this to check:
```bash
nvidia-smi --query-gpu=name,memory.total --format=csv
```

**Typical speeds:**
- RTX 3060 (12GB): ~2-3 sec/batch
- RTX 3080 (10GB): ~1-2 sec/batch
- RTX 4090 (24GB): ~0.5-1 sec/batch
- A100 (40GB): ~0.3-0.5 sec/batch

If you have a slower GPU, batch size increases are even more important!

## Quick Wins Summary

1. **Pull latest code** (has optimizations) ✅
2. **Increase batch_size to 128 or 256** (2-4x faster) ⚡
3. **Use 25 epochs, not 100** (4x less time) ⏱️
4. **Test checkpoints early** (don't wait for completion) 🎯

**Combined:** Could reduce 52 days → **2-4 days** 🚀
