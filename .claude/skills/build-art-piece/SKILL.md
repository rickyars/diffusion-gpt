---
name: build-art-piece
description: Build the WE art piece HTML file from ONNX model and scripts
allowed-tools: Bash(python:*), Read
---

# Build WE Art Piece

This skill builds the complete self-contained HTML file for the "WE" art installation.

## Prerequisites

- ONNX model must exist at `models/confessions_model_merged.onnx`
- Vocabulary must exist at `vocab/confessions_vocab.json`
- Build script at `scripts/art-piece/build.py`

## Steps

1. **Verify Prerequisites**
   ```bash
   ls models/confessions_model_merged.onnx vocab/confessions_vocab.json scripts/art-piece/build.py
   ```

2. **Run Build Script**
   ```bash
   python scripts/art-piece/build.py
   ```

3. **Verify Output**
   - Check that `scripts/art-piece/we.html` was created
   - Verify file size is under 100MB
   - Report the file size to the user

4. **Launch for Testing**
   ```bash
   start scripts/art-piece/we.html
   ```

## Expected Output

- HTML file: `scripts/art-piece/we.html`
- Size: ~60-65 MB (under 100MB budget)
- Features: Embedded ONNX model, inference engine, audio system, state machine

## Troubleshooting

- **Model not found**: Run `python scripts/art-piece/export_to_onnx.py` first
- **Merge needed**: Run `python scripts/art-piece/merge_onnx_data.py` if .onnx.data file exists
- **Build fails**: Check that all Python dependencies are installed
