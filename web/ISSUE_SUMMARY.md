# Issue Summary: ONNX Web Inference Debugging

## Current Status

### ✅ PyTorch Model Works
The `shakespeare.pt` model generates coherent text in PyTorch:
```bash
python test_pytorch_generation.py --model ../models/shakespeare.pt
# Output: Coherent text with structure, spaces, punctuation
```

### ❌ Web/ONNX Still Produces Gibberish
After exporting with correct `cond_dim=64`, the web inference still fails.

## Diagnostic Checklist

### Step 1: Verify ONNX Export Succeeded
```bash
cd web
python export_to_onnx.py --checkpoint ../models/shakespeare.pt

# Should show:
# - Inferred: cond_dim=64 (not 128!)
# - ONNX model exported successfully
# - Files created in demo/ directory
```

**Check the output:**
- [ ] Does it show `cond_dim=64`?
- [ ] Did it create `demo/model.onnx`?
- [ ] Is the file size reasonable (~11-50MB)?

```bash
ls -lh demo/
# Should show:
# - model.onnx (11-50MB)
# - vocab.json (~1KB)
# - metadata.json (~1KB)
```

### Step 2: Verify ONNX Model Matches PyTorch
```bash
# Test ONNX model directly in Python
python debug_python.py --model ../models/shakespeare.pt

# This will:
# 1. Test PyTorch inference
# 2. Test ONNX inference (if onnxruntime installed)
# 3. Compare outputs
```

**What to look for:**
- PyTorch logits std should be > 1.0 (good)
- ONNX logits std should also be > 1.0 (good)
- PyTorch and ONNX outputs should match closely

### Step 3: Check Browser Cache
The browser might be using the **old cached ONNX model**!

**Solution:**
1. **Hard refresh** the page: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
2. Or **clear browser cache** for localhost
3. Or **use Incognito/Private mode**

### Step 4: Verify Web Loading Correct Files
Open browser console (F12) and check:

```javascript
// In browser console, check what files are loading
console.log('Model URL:', document.getElementById('modelUrl').value)
console.log('Vocab URL:', document.getElementById('vocabUrl').value)
console.log('Metadata URL:', document.getElementById('metadataUrl').value)
```

Default URLs in `index.html`:
- `./demo/model.onnx`
- `./demo/vocab.json`
- `./demo/metadata.json`

**Verify:**
- [ ] Are these the files you just exported?
- [ ] Check the file timestamps: `ls -la demo/`

### Step 5: Test with Quick Test Page

```bash
python test_server.py
# Open http://localhost:8000/quick_test.html
```

**Expected results:**
- **Before fix:** Std Dev: 0.0238 (< 0.1) ❌
- **After fix:** Std Dev: > 1.0 ✅

If quick_test still shows low std dev:
1. Old ONNX model is still being used (cache issue)
2. ONNX export failed silently
3. ONNX model is in wrong location

## Common Issues

### Issue A: Browser Cache
**Symptom:** Quick test shows low std (0.02-0.05)
**Solution:** Hard refresh or clear cache

### Issue B: Wrong ONNX File
**Symptom:** `demo/model.onnx` timestamp is old
**Solution:** Re-run export script, check for errors

### Issue C: ONNX Export Failed
**Symptom:** Export script showed errors but continued
**Solution:** Check export script output carefully for error messages

### Issue D: Model Path Wrong
**Symptom:** Export script loaded wrong checkpoint
**Solution:** Verify `--checkpoint` path points to `shakespeare.pt`

## Debugging Commands

```bash
# 1. Test PyTorch model (should work)
cd web
python test_pytorch_generation.py --model ../models/shakespeare.pt

# 2. Export ONNX (watch for errors!)
python export_to_onnx.py --checkpoint ../models/shakespeare.pt

# 3. Check files were created
ls -lh demo/
stat demo/model.onnx  # Check timestamp

# 4. Test ONNX vs PyTorch (need onnxruntime)
pip install onnxruntime
python debug_python.py --model ../models/shakespeare.pt

# 5. Start server and test in browser
python test_server.py
# Open http://localhost:8000/quick_test.html
```

## What to Report

If still having issues, please provide:

1. **Export output:**
   ```bash
   python export_to_onnx.py --checkpoint ../models/shakespeare.pt 2>&1 | tee export_log.txt
   ```

2. **File listing:**
   ```bash
   ls -lh demo/
   ```

3. **Quick test results:**
   - What std dev does quick_test.html show?
   - Screenshot of the output

4. **Browser console errors:**
   - Open DevTools (F12) → Console tab
   - Any red error messages?

5. **Metadata check:**
   ```bash
   cat demo/metadata.json
   # Should show "cond_dim": 64
   ```

## Expected Workflow

```
1. Export ONNX:
   python export_to_onnx.py --checkpoint ../models/shakespeare.pt
   ✓ Shows cond_dim=64
   ✓ Creates demo/model.onnx

2. Clear browser cache or hard refresh

3. Start server:
   python test_server.py

4. Test quick diagnostic:
   http://localhost:8000/quick_test.html
   ✓ Shows std > 1.0
   ✓ Shows meaningful token probabilities

5. Test full generation:
   http://localhost:8000/index.html
   ✓ Loads model successfully
   ✓ Generates coherent text after 64 steps
```

## Next Steps

Run these commands and share the output:

```bash
cd web

# 1. Export (copy all output)
python export_to_onnx.py --checkpoint ../models/shakespeare.pt

# 2. Check metadata
cat demo/metadata.json

# 3. Check file timestamps
ls -la demo/
```

This will tell us if the ONNX export is actually working correctly.
