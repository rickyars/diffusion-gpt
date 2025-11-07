# ONNX Web Inference Debugging Guide

This guide will help you debug the ONNX web inference when outputs are garbage.

## Problem

The web version loads an ONNX model and produces garbage text instead of coherent output. This could be due to:

1. **Model conversion issues** - ONNX export may have introduced errors
2. **Inference implementation** - JavaScript inference code may differ from Python
3. **Numerical precision** - Float32 vs Float64, or overflow/underflow issues
4. **Algorithm implementation** - Staggered score, transition kernel, or sampling issues
5. **Input preprocessing** - Incorrect tensor shapes or data types
6. **Vocabulary mismatch** - Wrong character mappings

## Tools Provided

### 1. `debug.html` - Interactive Web Debugger

A comprehensive debugging interface that:
- Loads and tests the ONNX model
- Logs all intermediate values
- Shows statistics at each step
- Compares distributions
- Displays generated text

**Features:**
- Single inference test (check model outputs)
- Single denoising step test (check algorithm)
- Full generation (end-to-end test)
- Detailed logging for every position
- Distribution statistics (min, max, mean, std)

### 2. `debug_python.py` - Python Reference Script

Generates reference outputs from Python/PyTorch:
- Tests PyTorch model
- Tests ONNX model (via onnxruntime)
- Compares outputs
- Tests one denoising step
- Saves detailed JSON outputs

**Outputs:**
- `debug_pytorch_output.json` - PyTorch inference results
- `debug_onnx_output.json` - ONNX inference results
- `debug_step_output.json` - Detailed denoising step data

## Step-by-Step Debugging Process

### Step 1: Generate Reference Data

```bash
cd web
python debug_python.py
```

This will:
1. Load the PyTorch model
2. Run test inference
3. Export ONNX and test it
4. Compare outputs
5. Test one complete denoising step
6. Save all data to JSON files

**What to look for:**
- PyTorch and ONNX outputs should match (within ~1e-5)
- Logits should be in range [-10, 10] typically
- Scores (exp(logits)) should be positive
- Transition probabilities should sum to 1.0
- Final probabilities should be positive and sum to ~1.0

### Step 2: Test Web Inference

```bash
python test_server.py
```

Open http://localhost:8000/debug.html

1. **Load Model** - Verify all files load successfully
2. **Test Single Inference**:
   - Check logits range (should be similar to Python)
   - Check for NaN or Inf values
   - Compare with `debug_pytorch_output.json`
3. **Test One Denoising Step**:
   - Compare timestep parameters (t, sigma, delta_sigma)
   - Compare scores at position 0
   - Compare transition probabilities
   - Compare final probabilities
   - Check how many tokens changed (should be ~5-20%)
4. **Run Full Generation**:
   - Watch progress through all 64 steps
   - Check if text becomes more coherent or stays random

### Step 3: Compare Outputs

Open browser console (F12) and compare:

**JavaScript (browser console):**
```javascript
// After running "Test Single Inference"
// Check the console logs
```

**Python (terminal):**
```bash
# View the saved outputs
cat debug_pytorch_output.json | head -50
cat debug_step_output.json | head -50
```

**Key comparisons:**
1. Input IDs - Should be the same (if using same seed)
2. Logits - Should match closely (< 1e-3 difference)
3. Scores - Should match after exp()
4. Staggered scores - Should match
5. Transition probs - Should match
6. Final probs - Should match
7. Sampling - Will differ (random) but distribution should be similar

### Step 4: Identify the Issue

#### Issue A: Model Output is Wrong

**Symptoms:**
- Logits are all the same
- Logits are NaN or Inf
- Logits are in wrong range
- PyTorch and ONNX outputs differ significantly

**Solutions:**
- Re-export ONNX model with different opset version
- Check model inputs (dtype, shape)
- Verify sigma value is passed correctly
- Check if model layers are supported in ONNX

#### Issue B: Staggered Score is Wrong

**Symptoms:**
- Staggered scores are negative
- Staggered scores are NaN or Inf
- Staggered scores differ from Python

**Check:**
```javascript
// In staggeredScore function
const expFactor = Math.exp(-deltaSigma);
const scoreSum = score.reduce((a, b) => a + b, 0);
const correction = ((expFactor - 1) / (vocabSize * expFactor)) * scoreSum;
```

**Common issues:**
- Integer division instead of float
- Wrong sign on correction term
- Wrong order of operations

#### Issue C: Transition Kernel is Wrong

**Symptoms:**
- Transition probabilities don't sum to 1.0
- Diagonal is wrong
- All transitions are uniform

**Check:**
```javascript
// In computeTransition function
const baseProb = (1 - Math.exp(-deltaSigma)) / vocabSize;
// ...
const diagFill = 1 - trans.reduce((a, b) => a + b, 0);
trans[currentIdx] = diagFill;
```

**Common issues:**
- Off-by-one errors in vocabulary size
- Wrong calculation of diagonal element
- Missing normalization

#### Issue D: Sampling is Wrong

**Symptoms:**
- Always samples the same token
- Samples invalid tokens
- Distribution is wrong

**Check:**
```javascript
// In sampleCategorical function
const sum = probs.reduce((a, b) => a + b, 0);
const normalized = probs.map(p => p / sum);
```

**Common issues:**
- Probability sum is 0 or NaN
- Random number generator not working
- Wrong cumulative sum calculation

#### Issue E: Geometric Noise is Wrong

**Symptoms:**
- Sigma values are wrong
- Text doesn't denoise properly
- Too many or too few tokens change

**Check:**
```javascript
function geometricNoise(t) {
    const barSigma = Math.pow(sigmaMin, 1 - t) * Math.pow(sigmaMax, t);
    return barSigma;
}
```

**Common issues:**
- Wrong sigma_min or sigma_max values
- Wrong t calculation
- Off-by-one in step counting

### Step 5: Common Issues Checklist

- [ ] Model loads without errors
- [ ] Vocabulary matches (check first 10 characters)
- [ ] Metadata is correct (vocab_size, block_size)
- [ ] Input tensors have correct dtype (int64 for IDs, float32 for sigma)
- [ ] Logits output has correct shape [1, seq_len, vocab_size]
- [ ] Logits are in reasonable range (typically -10 to 10)
- [ ] No NaN or Inf in any computation
- [ ] exp() doesn't overflow (scores should be finite)
- [ ] Probabilities are positive and sum to ~1.0
- [ ] RNG produces different values on each call
- [ ] Timestep calculation matches Python exactly
- [ ] Delta sigma is positive and decreasing

## Advanced Debugging

### Enable Verbose Logging

In `debug.html`, check:
- ✅ Verbose logging
- ✅ Log distribution stats
- ✅ Log each position (warning: slow!)

This will log every computation in detail.

### Compare Step-by-Step

1. Set the same seed in Python and JavaScript:
   ```python
   # Python
   torch.manual_seed(42)
   ```

   ```javascript
   // JavaScript
   rng = new SeededRandom(42);
   ```

2. Run single step in both:
   - Python: `python debug_python.py`
   - Web: Click "Test One Denoising Step" in `debug.html`

3. Compare outputs position by position

### Export Intermediate Values

Modify the code to save intermediate values:

```javascript
// Add after each computation
console.log('Scores:', scores[0].slice(0, 10));
console.log('Staggered:', stagScore.slice(0, 10));
console.log('Transition:', trans.slice(0, 10));
console.log('Probs:', probs.slice(0, 10));
```

### Visual Inspection

Look at the generated text:
- **Pure random** - Model not working at all
- **Repeated patterns** - Sampling issue
- **Mostly unchanged** - Transition or delta_sigma issue
- **Changes but nonsense** - Model or scoring issue
- **Gradually improves** - Algorithm working correctly!

## Expected Behavior

### During Denoising:

**Step 0-10:** (high noise, σ ≈ 20)
- Text should be very noisy, mostly random
- 10-30% of tokens should change each step
- Some structure might appear

**Step 10-30:** (medium noise, σ ≈ 5-10)
- Text should show more structure
- Common bigrams appear (th, he, in, er)
- 5-15% of tokens change each step

**Step 30-50:** (low noise, σ ≈ 1-5)
- Text should be readable
- Words form correctly
- 2-10% of tokens change each step

**Step 50-64:** (very low noise, σ ≈ 0.01-1)
- Text should be coherent
- Minor refinements
- 1-5% of tokens change each step

### Final Output:

Should look like Shakespeare-style text:
```
What have I done? The king is dead.
I cannot bear this burden any more,
For what is life without the one we love?
```

Not like:
```
xK3#mzQqR$*vW2j...  # Pure garbage - model broken
aaaaaaaaaaaaaaaa...  # Repeated chars - sampling broken
The The The The...  # Stuck - transition broken
```

## Files Reference

- `debug.html` - Interactive web debugger
- `debug_python.py` - Python reference script
- `index.html` - Production web demo (broken)
- `export_to_onnx.py` - Export PyTorch to ONNX
- `test_inference.py` - Original test script

## Next Steps After Identifying Issue

1. **Fix the bug** in the appropriate file:
   - Model issue → re-export with `export_to_onnx.py`
   - Algorithm issue → fix in `index.html` or `debug.html`
   - Data issue → check vocabulary and metadata files

2. **Verify the fix**:
   - Re-run `debug_python.py`
   - Re-test in `debug.html`
   - Compare outputs again

3. **Test production**:
   - Copy fix from `debug.html` to `index.html`
   - Test in `index.html`
   - Verify full generation works

4. **Document the issue**:
   - Note what was wrong
   - Note how you fixed it
   - Update this guide if needed

## Getting Help

If you're still stuck:

1. Save the debug outputs:
   ```bash
   # Python outputs
   cat debug_pytorch_output.json > debug_report.txt
   cat debug_onnx_output.json >> debug_report.txt
   cat debug_step_output.json >> debug_report.txt
   ```

2. Save browser console output:
   - Open browser console (F12)
   - Right-click → Save as... → debug_browser.log

3. Share:
   - The debug outputs
   - Browser console logs
   - Description of what you see vs. what you expect

Good luck debugging! 🐛🔍
