# Quantization-Aware Training (QAT) Guide

## What is QAT?

Quantization-Aware Training (QAT) trains your model with INT8 quantization awareness from the start, producing smaller and faster models than post-training quantization.

### Key Concept: Fake Quantization vs True Quantization

**During training (Fake Quantization):**
- Weights stored as FP32 (32 bits)
- Forward pass simulates INT8 behavior (rounds to INT8 range)
- Backward pass uses FP32 gradients (INT8 can't represent small values like 0.0001)
- Checkpoint files remain ~45 MB
- Model learns to work within INT8 constraints

**After training (True Quantization):**
- Run conversion: `model.convert_to_quantized()`
- Weights stored as INT8 (8 bits)
- ONNX export creates ~11 MB file
- 4x size reduction happens here

**Why two steps?** You can't train with true INT8 because gradients would be too coarse. Fake quantization lets the model learn INT8-friendly behavior while keeping gradient precision for learning.

## Benefits

- **4x smaller model**: 45 MB → 11 MB
- **~2x faster inference**: 640ms → ~320ms per step (browser)
- **Better quality**: Model learns INT8 constraints during training (vs post-training quantization which fails)
- **No gibberish**: Avoids dynamic range compression issues

## Usage

### 1. Enable QAT in config.yaml

```yaml
training:
  epochs: 50
  batch_size: 352
  learning_rate: 0.0001

  # Enable QAT
  use_qat: true  # Train with INT8 quantization awareness
  qat_backend: 'fbgemm'  # 'fbgemm' for x86, 'qnnpack' for ARM
```

### 2. Train normally

```bash
python scripts/training/train.py --config config.yaml --dataset confessions-sample
```

Training takes 24-48 hours (same as FP32). Checkpoints are saved with `qat_trained: true` marker.

### 3. Export to ONNX

```bash
python scripts/art-piece/export_to_onnx.py \
    --model models/confessions-sample.pt \
    --dataset confessions-sample
```

The export script automatically:
- Detects QAT checkpoint (`qat_trained: true`)
- Converts fake INT8 → true INT8
- Exports to ONNX (~11 MB)
- Skips post-training quantization (already quantized)

### 4. Test in browser

```bash
python scripts/art-piece/update_model.py --dataset confessions-sample --force
```

Open `scripts/art-piece/we.html` and check console for performance profiling.

## Expected Results

With 11.67M parameter model:
- Model size: 45 MB → 11 MB (4x reduction)
- Inference: 640ms → ~320ms per step (2x speedup)
- Quality: Same as FP32 (no gibberish)

**Note**: Won't hit 100ms/step target. For that, train smaller model (4 layers, 256 embedding) + QAT to get ~80ms/step.

## Troubleshooting

### Loss explodes or NaN
Lower learning rate in config.yaml:
```yaml
training:
  learning_rate: 0.00005  # Half of normal
```

### Output still gibberish
Check that QAT was actually enabled:
1. Did you see "QUANTIZATION-AWARE TRAINING ENABLED" at start?
2. Does checkpoint have `qat_trained: true`?
3. Did export script detect QAT model?

### Speedup less than 2x
INT8 speedup varies by hardware:
- Better on newer CPUs with AVX-512
- Better on ARM with NEON
- Limited on old hardware

## Resuming Training

If interrupted, resume with:
```bash
python scripts/training/train.py \
    --config config.yaml \
    --dataset confessions-sample \
    --resume models/confessions-sample_interrupted_epoch_X.pt
```

The script auto-detects QAT checkpoints and continues QAT training.

## Implementation Details

QAT integration is minimal (~120 lines total):
- `scripts/training/model_quantized.py` - QAT wrapper (50 lines)
- `scripts/training/train.py` - QAT support (50 lines)
- `scripts/art-piece/export_to_onnx.py` - QAT detection (20 lines)
- `config.yaml` - Two options (2 lines)

The wrapper uses PyTorch's built-in `torch.quantization` API with FakeQuantize modules.
