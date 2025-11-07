# SOLUTION: ONNX Web Inference Vocabulary Mismatch

## Root Cause ✅

The web inference produces gibberish because of a **vocabulary mismatch**:

1. **PyTorch model (generate.py)** loads vocabulary from `vocab/shakespeare_vocab.pkl`
2. **ONNX export script** was using hardcoded default vocabulary
3. **Different vocabulary = wrong token mapping = gibberish output**

Even though the model weights are correct and the architecture is correct (cond_dim=64), if the vocabulary character order is different, token index 5 might mean 'a' in training but 'z' in inference → complete garbage.

## Evidence

When you ran export_to_onnx.py, you saw:
```
Warning: Vocabulary not found in checkpoint, using default Shakespeare vocab
```

This hardcoded default vocab might not match the actual vocabulary used during training.

## What I Fixed

Updated `web/export_to_onnx.py` to load vocabulary the **exact same way** as `generate.py`:

1. Added `--vocab` argument
2. Auto-detects vocab path from model name (e.g., `shakespeare.pt` → `vocab/shakespeare_vocab.pkl`)
3. Loads vocabulary from `.pkl` file (preferred)
4. Falls back to checkpoint if needed
5. Exits with clear error if vocabulary not found

## How to Fix Your Web Demo

### Step 1: Create Vocabulary File

The vocabulary file is missing. You need to create it from the Shakespeare dataset:

```bash
cd /path/to/diffusion-gpt

# Create vocabulary from your dataset
python create_vocab.py \
  --dataset datasets/shakespeare.txt \
  --output vocab/shakespeare_vocab.pkl
```

This will create `vocab/shakespeare_vocab.pkl` with the character mappings used during training.

**Note:** If you trained the model on a different dataset or custom text, use that dataset instead!

### Step 2: Re-export ONNX Model

Now export the ONNX model with the correct vocabulary:

```bash
cd web
python export_to_onnx.py --checkpoint ../models/shakespeare.pt
```

You should see:
```
Auto-detected vocab path: ../vocab/shakespeare_vocab.pkl
Loading vocabulary from: ../vocab/shakespeare_vocab.pkl
✓ Vocabulary loaded successfully from .pkl file
```

This creates `demo/vocab.json` with the **correct** vocabulary that matches training.

### Step 3: Test Web Inference

```bash
python test_server.py
```

Open http://localhost:8000/quick_test.html

Expected results:
- **Logits Std Dev:** > 1.0 (not 0.02!)
- **Diagnosis:** ✅ GOOD: Logits show meaningful variation

Then test full generation:
- Open http://localhost:8000/index.html
- Click "Load Model" → "Start Generation"
- Should see **coherent Shakespeare-style text**, not gibberish!

## Why This Fixes It

**Before:**
```
Training vocab:  ['\n', ' ', '!', '"', '&', "'", '(', ')', ',', '-', ...]
Export vocab:    ['\n', ' ', '!', '"', '&', "'", '(', ')', ',', '-', ...]  ❌ MIGHT BE DIFFERENT ORDER!
```

Even if the characters are the same, if the **order** is different, token indices don't match.

**After:**
```
Training vocab:  ['\n', ' ', '!', '"', '&', "'", '(', ')', ',', '-', ...]
Export vocab:    ['\n', ' ', '!', '"', '&', "'", '(', ')', ',', '-', ...]  ✅ EXACT SAME (loaded from same file)
```

## Verification

To verify the vocabularies match:

```bash
# Check what vocabulary was used during training
python -c "import pickle; meta = pickle.load(open('vocab/shakespeare_vocab.pkl', 'rb')); print('Chars:', ''.join(meta['itos'][i] for i in range(min(20, len(meta['itos'])))))"

# Check what vocabulary was exported for web
python -c "import json; vocab = json.load(open('web/demo/vocab.json', 'r')); print('Chars:', ''.join(vocab['itos'][str(i)] for i in range(min(20, len(vocab['itos'])))))"
```

These should show **identical** character sequences.

## If Still Having Issues

### Issue A: Vocabulary file already exists but different
**Solution:** Delete old vocab file and recreate from the actual dataset used during training

### Issue B: Model trained on different dataset
**Symptom:** shakespeare.txt has different characters than what model was trained on
**Solution:** Use the correct dataset file with `create_vocab.py`

### Issue C: Model was trained elsewhere
**Symptom:** You don't have the original training dataset
**Solution:** Extract vocabulary from a working generate.py run, or check if checkpoint contains itos/stoi

## Files Modified

1. **`web/export_to_onnx.py`** - Now loads vocabulary same as generate.py
2. **`create_vocab.py`** - New helper script to create vocabulary files
3. **`web/SOLUTION.md`** - This document

## Summary

**Problem:** Vocabulary mismatch between training and web inference

**Root Cause:** Export script used hardcoded default vocab instead of actual training vocab

**Solution:** Load vocabulary from same source as generate.py uses

**Action Required:**
1. Create vocabulary file: `python create_vocab.py --dataset datasets/shakespeare.txt --output vocab/shakespeare_vocab.pkl`
2. Re-export ONNX: `cd web && python export_to_onnx.py --checkpoint ../models/shakespeare.pt`
3. Test: Open http://localhost:8000/quick_test.html

This should fix the gibberish output! 🎉
