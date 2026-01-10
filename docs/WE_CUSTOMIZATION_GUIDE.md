# WE Art Piece - Customization Guide

This guide explains how to customize the visual effects, timing, and flow of the WE art piece (`scripts/art-piece/we.html`), and how to convert PyTorch models to ONNX format for browser deployment.

---

## ONNX Conversion for Browser Deployment

To run the WE art piece in a browser, you need to convert your trained PyTorch model to ONNX format. This section explains the conversion process.

### Prerequisites

Ensure you have trained a model (see main README for training instructions):
```bash
python scripts/training/train.py --dataset confessions
```

This will create a checkpoint like `models/confessions.pt` or `models/confessions_epoch_25.pt`.

### Step 1: Export PyTorch Model to ONNX

Use the export script to convert your model:

```bash
python scripts/art-piece/export_to_onnx.py \
  --model models/confessions_epoch_25.pt \
  --dataset confessions
```

This will create:
- `models/confessions_model.onnx` - The ONNX model file
- `models/confessions_model_quantized.onnx` - INT8 quantized version (if quantization succeeds)
- `vocab/confessions_vocab.json` - Character vocabulary in JSON format

**What it does:**
- Converts PyTorch model to ONNX format (opset version 18)
- Performs INT8 quantization to reduce file size (typically 30-50% smaller)
- Falls back to non-quantized ONNX if quantization fails
- Exports vocabulary from pickle to JSON format
- Prints file sizes and size reduction percentages

**Options:**
- `--model`: Path to your trained PyTorch checkpoint (required)
- `--dataset`: Dataset name used for output files (required)

### Step 2: Merge External ONNX Data (If Needed)

If the ONNX export created a `.onnx.data` file, merge it into a single file:

```bash
python scripts/art-piece/merge_onnx_data.py \
  --input models/confessions_model.onnx
```

This creates `models/confessions_model_merged.onnx` with all data embedded.

**When is this needed?**
- Large models may have external data files to reduce file size during creation
- Browser deployment requires a single ONNX file
- Merging is done automatically if a `.data` file exists

**Options:**
- `--input`: Path to ONNX model with external data (required)
- `--output`: Output path (optional, defaults to `{input}_merged.onnx`)

### Step 3: Build the HTML Art Piece

Once you have the ONNX model, use the build script to create the final HTML file:

```bash
python scripts/art-piece/build.py --dataset confessions
```

This will create `scripts/art-piece/we.html` (~60MB) with:
- Embedded ONNX model (quantized)
- Embedded vocabulary
- All JavaScript code, shaders, and audio
- Self-contained (no external dependencies)

**Options:**
- `--dataset`: Dataset name to find the ONNX model (required)
- `--model`: Custom ONNX model path (optional)

### Complete Workflow Example

Here's a complete example from training to HTML:

```bash
# 1. Train the model
python scripts/training/train.py --dataset confessions

# 2. Export to ONNX
python scripts/art-piece/export_to_onnx.py \
  --model models/confessions_epoch_25.pt \
  --dataset confessions

# 3. Merge ONNX data (if needed)
python scripts/art-piece/merge_onnx_data.py \
  --input models/confessions_model.onnx

# 4. Build HTML art piece
python scripts/art-piece/build.py --dataset confessions

# 5. View in browser
start scripts/art-piece/we.html
```

### File Size Optimization

The export process optimizes for browser deployment:

| Component | Size | Notes |
|-----------|------|-------|
| ONNX model (quantized) | 15-25 MB | INT8 quantization reduces size 30-50% |
| Vocabulary | <1 KB | Character mappings (tiny) |
| JavaScript code | ~50 KB | All logic and utilities |
| ONNX Runtime Web | ~2.5 MB | Included in final HTML |
| CRT Shaders & Audio | ~20 KB | Phosphor glow, scanlines, 60Hz hum |
| **Total** | **~60 MB** | Single self-contained HTML file |

### Troubleshooting ONNX Conversion

**Model fails to load during export:**
- Ensure model checkpoint exists at the specified path
- Check that the dataset name matches your training

**Quantization fails:**
- The script falls back to non-quantized ONNX automatically
- Non-quantized models are still browser-compatible, just larger (~30-50% bigger)
- Ensure `onnxruntime` is installed: `pip install onnxruntime`

**ONNX file is very large (>100MB):**
- Quantization may not have worked—check console output
- Try with a smaller model architecture (reduce `n_layer`, `n_embd` in config.yaml)

**"Model not found" during build:**
- Ensure you ran the export_to_onnx.py script first
- Check that the dataset name matches (case-sensitive)
- Verify files exist: `ls models/confessions_model*.onnx`

---

## Quick Reference

### Configuration Constants

At the top of the file (around lines 489-505), you'll find the main configuration:

```javascript
// Diffusion configuration
const DIFFUSION_CONFIG = {
  TOTAL_STEPS: 128,      // Total denoising steps (quality)
  DISPLAY_FRAMES: 64,    // Frames to actually display (performance)
  // Frame skip = TOTAL_STEPS / DISPLAY_FRAMES (e.g., 128/64 = show every 2nd frame)
};

const TIMING = {
  VOID: 1000,              // Initial pause (ms)
  WE_FADEIN: 1500,         // WE fade in duration
  WE_HOLD: 2500,           // WE display time
  WE_DISSOLVE: 2500,       // WE dissolve/glitch duration
  CONF_GENERATING: 500,    // Buffer loading state
  CONF_DIFFUSE: 4500,      // Diffusion animation playback (controls FPS)
  CONF_HOLD: 25000,        // Confession reading time
  CONF_FADE: 6000          // Confession fade out duration
};
```

---

## State Flow

The art piece follows this state machine:

```
VOID → WE_FADEIN → WE_HOLD → WE_DISSOLVE → CONF_GENERATING → CONF_DIFFUSE → CONF_HOLD → CONF_FADE → (loops back)
```

### State Descriptions

| State | What Happens | Visual |
|-------|-------------|--------|
| `VOID` | Black screen pause | Nothing visible |
| `WE_FADEIN` | WE fades in | WE text appears gradually |
| `WE_HOLD` | WE displays with glitch effects | Big WE with RGB split glitches |
| `WE_DISSOLVE` | WE dissolves into random characters | Characters randomize and fade |
| `CONF_GENERATING` | Buffering next confession | Pulsing WE with blinking cursor |
| `CONF_DIFFUSE` | Diffusion animation plays back | Text emerges from noise |
| `CONF_HOLD` | Confession fully visible | Readable text with subtle glitch |
| `CONF_FADE` | Confession fades out | Text gradually disappears |

### Reordering States

To change the flow, modify the state transitions in the `updateState()` method (lines ~620-694):

**Example: Skip WE_DISSOLVE**
```javascript
case States.WE_HOLD:
  if (this.stateTime > TIMING.WE_HOLD) {
    this.transitionTo(States.CONF_GENERATING);  // Skip directly to generation
  }
  break;
```

**Example: Loop back to WE after each confession**
```javascript
case States.CONF_FADE:
  if (this.stateTime > TIMING.CONF_FADE) {
    this.transitionTo(States.WE_FADEIN);  // Go back to WE instead of looping confessions
  }
  break;
```

---

## Diffusion Settings

### Quality vs Performance

**TOTAL_STEPS**: How many denoising steps the model runs (quality)
- Higher = better quality, slower generation
- Lower = faster generation, lower quality
- Default: 128

**DISPLAY_FRAMES**: How many frames to actually show (performance)
- The system will skip frames: shows every `TOTAL_STEPS / DISPLAY_FRAMES` steps
- Default: 64 (shows every 2nd frame)

**Examples:**
```javascript
// Maximum quality and smoothness
TOTAL_STEPS: 128,
DISPLAY_FRAMES: 128,  // Shows all frames

// High quality, moderate playback
TOTAL_STEPS: 128,
DISPLAY_FRAMES: 64,   // Shows every 2nd frame

// Ultra quality, selective playback
TOTAL_STEPS: 256,
DISPLAY_FRAMES: 64,   // Shows every 4th frame

// Fast generation
TOTAL_STEPS: 32,
DISPLAY_FRAMES: 32,   // Shows all frames
```

### Playback Speed (FPS)

The `CONF_DIFFUSE` timing controls how fast frames play back:

```javascript
CONF_DIFFUSE: 4500,  // 4.5 seconds for 64 frames = ~14 FPS
```

**Formula:** `CONF_DIFFUSE = (DISPLAY_FRAMES / desired_FPS) * 1000`

**Examples:**
```javascript
// 24 FPS (smooth, cinematic)
CONF_DIFFUSE: 2667,  // 64 frames / 24 FPS * 1000

// 30 FPS (very smooth)
CONF_DIFFUSE: 2133,  // 64 frames / 30 FPS * 1000

// 10 FPS (slow, deliberate)
CONF_DIFFUSE: 6400,  // 64 frames / 10 FPS * 1000
```

---

## Visual Effects

### WE Text Glitch Effects

Located in `renderWE()` function (lines ~869-908):

**Jitter Intensity** (line 877):
```javascript
const glitchIntensity = this.state === States.WE_HOLD ? 0.005 : 0;
```
- Increase for more jitter: `0.01`, `0.02`
- Decrease for less: `0.002`, `0.001`
- Set to `0` to disable

**RGB Channel Split Frequency** (line 895):
```javascript
if (this.state === States.WE_HOLD && Math.random() < 0.05) {
```
- `0.05` = 5% chance per frame
- Increase for more frequent: `0.1` (10%), `0.2` (20%)
- Decrease for rarer: `0.02` (2%), `0.01` (1%)

**RGB Split Distance** (lines 898, 901):
```javascript
this.textCtx.fillText('WE', w / 2 - 2 + glitchX, h / 2 + glitchY);  // Red offset
this.textCtx.fillText('WE', w / 2 + 2 + glitchX, h / 2 + glitchY);  // Blue offset
```
- Change `2` to larger number for wider split: `5`, `10`
- Change to smaller for tighter split: `1`, `0.5`

**Adding glitch to other states:**

To add glitch to `CONF_GENERATING` (WE with cursor):
```javascript
// Line 877
const glitchIntensity = (this.state === States.WE_HOLD || this.state === States.CONF_GENERATING) ? 0.005 : 0;

// Line 895
if ((this.state === States.WE_HOLD || this.state === States.CONF_GENERATING) && Math.random() < 0.05) {
```

### Confession Text Glitch Effects

Located in `renderConfession()` function (lines ~916-951):

**Jitter Intensity** (line 931):
```javascript
const glitchIntensity = 0.003;
```
- Increase for more jitter: `0.005`, `0.01`
- Decrease for less: `0.001`, `0.0005`

**RGB Split Frequency** (line 936):
```javascript
if (Math.random() < 0.05) {
```
- Same as WE text - adjust probability (0.0 to 1.0)

**RGB Split Distance** (lines 939, 943):
```javascript
this.renderConfessionText(text, w, h, fontSize, glitchX - 2, glitchY);  // Red
this.renderConfessionText(text, w, h, fontSize, glitchX + 2, glitchY);  // Blue
```
- Adjust the `2` value to change offset distance

---

## CRT Shader Effects

The CRT effects are in the fragment shader (lines ~1040-1217). Here are the main controls:

### Barrel Distortion

**Strength** (line 1069):
```javascript
vec2 distortedUV = barrelDistortion(scaledUV, 0.25);
```
- Increase for more curvature: `0.3`, `0.4`
- Decrease for less: `0.15`, `0.1`

**Pre-scaling** (line 1068):
```javascript
vec2 scaledUV = (uv - 0.5) * 0.92 + 0.5;
```
- Decrease `0.92` to zoom in more (prevents edge clipping): `0.88`, `0.85`
- Increase to show more of the edges: `0.95`, `0.98`

### Scanlines

**Darkness** (line 1107):
```javascript
float scanlineEffect = mix(0.5, 1.0, scanlineMask);
```
- First value controls darkness: lower = darker scanlines
  - `0.3` = very dark scanlines
  - `0.7` = subtle scanlines

**Density** (line 1101):
```javascript
float scanlineCount = uResolution.y * 0.75;
```
- Increase multiplier for more scanlines: `1.0`, `1.5`
- Decrease for fewer: `0.5`, `0.25`

### Phosphor Glow

**Intensity** (lines 1135-1136):
```javascript
return color + tintedBloom * 4.0 + bloom * 3.0;
```
- Increase multipliers for stronger glow: `5.0`, `6.0`
- Decrease for subtler glow: `2.0`, `1.0`

**Green Tint** (line 1132):
```javascript
vec3 phosphorColor = vec3(0.2, 1.0, 0.3);
```
- Adjust RGB values (0.0 to 1.0)
- More green: `vec3(0.1, 1.0, 0.2)`
- Less green: `vec3(0.3, 1.0, 0.5)`

### Noise/Static

**Background Static Intensity** (line 1146):
```javascript
color += backgroundGlow * darkness * 0.5;
```
- Increase for more visible static: `0.8`, `1.0`
- Decrease for less: `0.3`, `0.1`

**Animated Noise Speed** (line 1143):
```javascript
float staticPattern = animatedNoise(uv * 3.0, uTime * 2.0);
```
- Increase `2.0` for faster animation: `3.0`, `4.0`
- Decrease for slower: `1.0`, `0.5`

### Vignette

**Strength** (line 1155):
```javascript
float vignette = 1.0 - length(vignetteUV) * 0.6;
```
- Increase `0.6` for stronger vignette: `0.8`, `1.0`
- Decrease for subtler: `0.4`, `0.2`

---

## Text Sizing

### WE Text Size

**Font size** (line 885):
```javascript
const fontSize = h * 0.2;  // 20% of screen height
```
- Increase for bigger WE: `0.25`, `0.3`
- Decrease for smaller: `0.15`, `0.1`

### Confession Text Size

**Font size** (line 924):
```javascript
const fontSize = Math.max(18, h * 0.028);
```
- Increase `0.028` for larger text: `0.035`, `0.04`
- Decrease for smaller: `0.02`, `0.015`
- The `Math.max(18, ...)` ensures minimum size

---

## Timing Examples

### Fast-paced Experience
```javascript
const TIMING = {
  VOID: 500,
  WE_FADEIN: 1000,
  WE_HOLD: 1500,
  WE_DISSOLVE: 1000,
  CONF_GENERATING: 500,
  CONF_DIFFUSE: 2000,   // Faster playback
  CONF_HOLD: 10000,     // Shorter reading time
  CONF_FADE: 3000
};
```

### Contemplative Experience
```javascript
const TIMING = {
  VOID: 2000,
  WE_FADEIN: 3000,
  WE_HOLD: 5000,
  WE_DISSOLVE: 3000,
  CONF_GENERATING: 500,
  CONF_DIFFUSE: 8000,   // Slower, more visible diffusion
  CONF_HOLD: 40000,     // Long reading time
  CONF_FADE: 10000
};
```

### Minimal (WE only, no confessions)
```javascript
// In updateState(), change WE_DISSOLVE to loop back:
case States.WE_DISSOLVE:
  if (this.stateTime > TIMING.WE_DISSOLVE) {
    this.transitionTo(States.VOID);  // Loop back to beginning
  }
  break;
```

---

## Common Customizations

### Remove a State

To skip `WE_DISSOLVE` entirely:
```javascript
case States.WE_HOLD:
  if (this.stateTime > TIMING.WE_HOLD) {
    this.transitionTo(States.CONF_GENERATING);  // Skip WE_DISSOLVE
  }
  break;
```

### Add VOID Between Each Cycle
```javascript
case States.CONF_FADE:
  if (this.stateTime > TIMING.CONF_FADE) {
    this.transitionTo(States.VOID);  // Pause before looping
  }
  break;
```

### Disable All Glitch Effects

**WE text** (line 877):
```javascript
const glitchIntensity = 0;  // No jitter
```

**WE RGB split** (line 895):
```javascript
if (false && this.state === States.WE_HOLD && Math.random() < 0.05) {  // Never triggers
```

**Confession glitch** (line 931):
```javascript
const glitchIntensity = 0;
```

---

## Troubleshooting

### Text is too small/large
- Adjust font size multipliers (see Text Sizing section)

### Animation is too fast/slow
- Adjust `CONF_DIFFUSE` timing for playback speed
- Adjust other `TIMING` values for state durations

### Not enough glitch
- Increase glitch probability from `0.05` to higher values
- Increase glitch intensity from `0.005` to higher values
- Increase RGB offset distances from `2` to higher values

### Too much glitch
- Decrease probabilities and intensities
- Set `glitchIntensity = 0` to disable completely

### CRT effects too strong
- Decrease shader multipliers in the fragment shader
- Reduce barrel distortion amount
- Lighten scanlines by adjusting mix values

### Generation taking too long
- Reduce `TOTAL_STEPS` from 128 to lower values (64, 32)
- The displayed quality is controlled by `DISPLAY_FRAMES`

---

## Advanced: Custom State Flow

You can create entirely custom flows by modifying the state transitions:

**Example: Alternate between WE and confessions**
```javascript
case States.CONF_FADE:
  if (this.stateTime > TIMING.CONF_FADE) {
    this.transitionTo(States.WE_FADEIN);  // Go back to WE
  }
  break;

case States.WE_DISSOLVE:
  if (this.stateTime > TIMING.WE_DISSOLVE) {
    this.transitionTo(States.CONF_GENERATING);  // Go to confession
  }
  break;
```

**Example: Multiple confessions before returning to WE**

Add a counter to the constructor:
```javascript
this.confessionCount = 0;
```

Then modify CONF_FADE:
```javascript
case States.CONF_FADE:
  if (this.stateTime > TIMING.CONF_FADE) {
    this.confessionCount++;
    if (this.confessionCount >= 3) {
      this.confessionCount = 0;
      this.transitionTo(States.VOID);  // Back to WE after 3 confessions
    } else {
      this.transitionTo(States.CONF_GENERATING);  // Another confession
    }
  }
  break;
```

---

## Reference: State Code Locations

| What | Line Range | Description |
|------|-----------|-------------|
| Configuration | ~489-505 | DIFFUSION_CONFIG and TIMING constants |
| State transitions | ~620-694 | updateState() method |
| WE rendering | ~869-908 | renderWE() with glitch effects |
| WE dissolve | ~910-946 | renderWEDissolve() with character randomization |
| Confession rendering | ~916-951 | renderConfession() with glitch effects |
| Confession with cursor | ~803-820 | CONF_GENERATING render case |
| CRT shader | ~1040-1217 | Fragment shader with all CRT effects |

---

## Questions?

For implementation details, see the main specification: `docs/we-spec.md`

For technical architecture and model info, see: `docs/WEB_DEMO_README.md`
