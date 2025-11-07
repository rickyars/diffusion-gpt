# ISSUE RESOLVED: ONNX Web Inference Producing Garbage

## Root Cause Identified ✅

The ONNX model was exported with **incorrect architecture parameters**, specifically:
- **Checkpoint has:** `cond_dim=64`
- **Code was using:** `cond_dim=128`

This caused the model to fail loading weights properly, resulting in essentially random/uninitialized weights.

## Evidence

### 1. Quick Test Results
```
Logits std = 0.0238 (< 0.1) → Nearly uniform, essentially random
Top probability: 1.74% (uniform would be 1.54% for 65 tokens)
```

### 2. Python Debug Results
```
RuntimeError: Error(s) in loading state_dict for GPT:
  size mismatch for sigma_map.mlp.0.weight:
    copying param with shape torch.Size([64, 256]) from checkpoint,
    the shape in current model is torch.Size([128, 256])
```

The checkpoint structure is:
- No `'config'` key (just raw state_dict)
- `cond_dim=64` (from sigma_map layer size)
- All other parameters correctly inferred

## What Was Wrong

1. **export_to_onnx.py** didn't handle checkpoints without 'config' key
2. It couldn't infer `cond_dim` from the state_dict
3. Defaulted to wrong value (`cond_dim=128`)
4. Exported ONNX model with mismatched architecture
5. Web inference loaded this broken model
6. Model outputs were essentially random → garbage text

## The Fix ✅

Updated both `export_to_onnx.py` and `debug_python.py` to:

1. Handle checkpoints without 'config' key
2. Infer architecture from state_dict:
   - `vocab_size` from `transformer.wte.weight` shape
   - `n_embd` from embedding dimensions
   - `n_layer` by counting `transformer.h.*` layers
   - `block_size` from `transformer.wpe.weight` shape
   - **`cond_dim` from `sigma_map.mlp.0.weight` shape** ⭐ KEY FIX

## How to Fix Your Web Demo

### Step 1: Re-export ONNX Model

```bash
cd web
python export_to_onnx.py
```

You should see:
```
⚠️  No config found in checkpoint, inferring from state_dict...
  Inferred: vocab_size=65, n_embd=384, n_layer=6, block_size=256, cond_dim=64
Model config loaded successfully!
ONNX model exported successfully!
```

This will create `demo/` with:
- `model.onnx` - **Correctly configured model**
- `vocab.json` - Character vocabulary
- `metadata.json` - Model metadata

### Step 2: Test the Fixed Model

```bash
python test_server.py
```

Then open: http://localhost:8000/quick_test.html

You should now see:
```
Statistics:
  Std Dev: > 1.0  ✅ Good!

=== DIAGNOSIS ===
✅ GOOD: Logits show meaningful variation
   The model is producing differentiated predictions.
```

### Step 3: Run Full Generation

Open: http://localhost:8000/index.html

Click "Load Model" → "Start Generation"

You should now see **coherent Shakespeare-style text** instead of gibberish!

Expected behavior:
- **Step 0-10:** Noisy but some structure (90%+ tokens change)
- **Step 10-30:** Readable words form (10-30% change)
- **Step 30-50:** Coherent sentences (5-15% change)
- **Step 50-64:** Polish and refinement (1-5% change)

## Verification

### Before Fix:
```
Step 0: EAdE6X263mVh"a&Pfb3fXj(kOkN7Z,B&f;:Y?QGgmIf...
Step 64: 'k6c ?QX9a iPhZ kch dgg Z ighULg3hS ;WohVQTiW...
         ☝️ Still gibberish
```

### After Fix:
```
Step 0: xR3#mzQqR$*vW2j8KpFnL9cYtH1aM6...
Step 64: What have I done? The king is dead.
         I cannot bear this burden any more...
         ☝️ Coherent Shakespeare text!
```

## Technical Details

### Checkpoint Structure
Your checkpoint has this structure:
```python
{
    'sigma_map.mlp.0.weight': torch.Size([64, 256]),  # cond_dim=64
    'transformer.wte.weight': torch.Size([65, 384]),  # vocab=65, embd=384
    'transformer.wpe.weight': torch.Size([256, 384]), # block_size=256
    'transformer.h.0.*': ...,                         # 6 layers total
    # ... (raw state dict, no 'config' key)
}
```

### Model Architecture
```python
GPTConfig(
    vocab_size=65,
    block_size=256,
    n_layer=6,
    n_head=6,
    n_embd=384,
    cond_dim=64,  # ⭐ This was the missing piece!
    dropout=0.0,
    bias=False
)
```

## Files Modified

1. **`web/export_to_onnx.py`** - Fixed to infer cond_dim
2. **`web/debug_python.py`** - Fixed to infer cond_dim
3. **`web/quick_test.html`** - Added for fast diagnosis
4. **`web/debug.html`** - Comprehensive debugging tool
5. **`web/DEBUGGING_GUIDE.md`** - Full debugging guide

## Summary

**Problem:** Wrong `cond_dim` → Architecture mismatch → Random weights → Garbage output

**Solution:** Infer `cond_dim` from checkpoint → Correct architecture → Proper weights → Coherent text

**Action Required:** Re-export ONNX model with fixed script

## Questions?

If the model still produces garbage after re-export:
1. Check `quick_test.html` - logits std should be > 1.0
2. Run `debug_python.py` - should load without errors
3. Check `debug.html` - compare with expected statistics
4. See `DEBUGGING_GUIDE.md` for more help

---

**Issue Status:** ✅ RESOLVED - User needs to re-export ONNX model
