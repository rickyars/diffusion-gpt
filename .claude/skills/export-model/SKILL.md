---
name: export-model
description: Export PyTorch model to ONNX format for browser deployment
allowed-tools: Bash(python:*), Read
---

# Export Model to ONNX

This skill exports a trained PyTorch discrete diffusion model to ONNX format for browser-based inference.

## Prerequisites

- Trained PyTorch model at `models/{model_name}.pt`
- Vocabulary pickle at `vocab/{model_name}_vocab.pkl`
- Export script at `scripts/art-piece/export_to_onnx.py`

## Steps

1. **Run Export Script**
   ```bash
   python scripts/art-piece/export_to_onnx.py
   ```

   The script will:
   - Load the PyTorch model
   - Export to ONNX format (opset 18)
   - Attempt quantization (falls back to full precision if it fails)
   - Export vocabulary to JSON format
   - Report file sizes

2. **Merge External Data** (if needed)

   If the export creates separate `.onnx` and `.onnx.data` files:
   ```bash
   python scripts/art-piece/merge_onnx_data.py
   ```

   This creates a single `models/{model_name}_merged.onnx` file.

3. **Verify Output**
   - `models/{model_name}.onnx` (or `{model_name}_merged.onnx`)
   - `vocab/{model_name}_vocab.json`
   - File sizes should be reasonable (< 50MB for model, < 5KB for vocab)

## Expected Behavior

- Export time: ~30 seconds
- Model size: ~45MB (non-quantized) or ~12MB (quantized)
- Quantization may fail on some model architectures - this is OK
- Vocabulary export should always succeed

## Troubleshooting

- **Model not found**: Check model path in `models/` directory
- **Quantization error**: Script will automatically fall back to non-quantized
- **Out of memory**: Export on a machine with more RAM or use smaller model
- **ONNX version error**: Install `pip install --upgrade onnx onnxruntime`
