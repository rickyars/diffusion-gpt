# WE — Technical Specification

## Overview

A self-contained HTML art piece for Claire Silver's AI Art Contest 9. The piece displays an eternal cycle of AI-generated confessions, trained on 1 million real anonymous confessions from Reddit. It runs forever without interaction. The viewer is a witness, not a participant.

**Deployment target:** IPFS or Arweave (permanent, immutable)
**File format:** Single self-contained HTML file (all assets inlined)
**Target size:** Under 15MB (ideally under 10MB)

---

## Concept

The screen is a void. The word **WE** appears, glowing like phosphor. It dissolves into noise. A confession emerges, character by character, fighting through static—the diffusion model denoising in real-time. The confession holds, then fades. **WE** returns. This loops eternally.

The viewer cannot interact. They can only attend.

---

## Visual Design

### Color Palette

- **Background:** Pure black (`#000000`)
- **Primary text:** Phosphor green (`#33ff33` or `#00ff66`)
- **Glow:** Same green, blurred
- **Scanlines:** Black with slight transparency

### Typography

- **Font:** VT323 (Google Font) or similar terminal/CRT font
- **Fallback:** monospace
- **Size:** Large enough to read comfortably, responsive to viewport
- **The font must be embedded/inlined** (base64 or woff2 inline)

### CRT Effect (The "Slap")

Achieve via WebGL post-processing shader. The effect should feel like a ghostly CRT floating in void—not a physical monitor, but the memory of one.

**Required effects:**

1. **Bloom/Glow**
   - Bright pixels bleed light into surrounding area
   - Gaussian blur of text composited additively underneath
   - Intensity: strong but not overwhelming

2. **Scanlines**
   - Horizontal lines across entire viewport
   - Subtle, not overpowering
   - Slight transparency variation (darker bands)
   - Should feel like they extend into the void, not contained by a screen

3. **Barrel Distortion (subtle)**
   - Very slight curvature, as if projected on a curved surface
   - Optional: can skip if it complicates things, but adds to the spectral CRT feel

4. **Chromatic Aberration**
   - Slight RGB split at edges
   - Very subtle—just enough to feel "analog"

5. **Vignette**
   - Darker at corners/edges
   - Soft falloff
   - Reinforces the void feeling

6. **Noise/Static**
   - Subtle animated grain over everything
   - Intensity increases during diffusion, settles when text resolves
   - Film grain quality, not harsh TV static

7. **Flicker**
   - Very subtle brightness variation (1-3%)
   - Slow, irregular
   - Makes it feel alive/unstable

### Text Animation

**The "WE" title:**
- Fades in from black (opacity 0 → 1 over ~2 seconds)
- Holds, glowing, breathing (subtle pulse in brightness)
- Dissolves: characters rapidly cycle through random glyphs, then scatter/fade

**Confession diffusion:**
- Starts as random characters (the charset used by the model)
- Characters "fight" to resolve—flickering between random and correct
- Use actual diffusion steps from the model if possible, or simulate:
  - Early steps: mostly noise, occasional correct character flashes
  - Middle steps: patches of coherence emerging
  - Late steps: almost readable, final characters snapping into place
- Resolved text glows steady
- Total duration: ~30 seconds

**Confession fade:**
- Characters decay (randomize then fade, or just fade)
- Glow lingers slightly after characters disappear
- Total duration: ~3-5 seconds

---

## Audio Design

Use Web Audio API. No external audio files (keeps it self-contained and small).

### Soundscape

1. **Low Hum (60Hz drone)**
   - Synthesized sine wave, ~60Hz
   - Very low volume, felt more than heard
   - Constant, with very slow volume modulation

2. **Electrical Crackle (occasional)**
   - Filtered noise bursts
   - Random timing, sparse
   - Subtle—like old electronics

3. **Static Texture**
   - Filtered white noise, very quiet
   - Increases slightly during diffusion phase
   - Decreases when text resolves

4. **Reverb**
   - Apply convolution reverb or algorithmic reverb to everything
   - Creates sense of space (the void has acoustics)
   - Long tail, dark character

### Audio Flow

1. Silence on page load
2. User clicks anywhere to enable audio (browser requirement)
3. Hum fades in over 3 seconds
4. Hum + subtle static continue throughout
5. Crackles trigger randomly (average 1 per 10-15 seconds)
6. Static swells slightly during diffusion, settles on resolve

**Note:** Audio should work without click, but modern browsers require user interaction. Add a subtle "click anywhere to enable audio" message that fades after interaction, or just let it be silent for those who don't click.

---

## Technical Architecture

### Rendering Pipeline

```
┌─────────────────────────────────────────────────┐
│  Canvas 2D (text rendering)                      │
│  - Render "WE" or confession text               │
│  - Render diffusion animation (character swap)  │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  WebGL Post-Processing                           │
│  - Bloom pass                                    │
│  - Scanlines                                     │
│  - Barrel distortion                             │
│  - Chromatic aberration                          │
│  - Vignette                                      │
│  - Noise overlay                                 │
│  - Flicker                                       │
└────────────────────┬────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│  Screen Output                                   │
└─────────────────────────────────────────────────┘
```

### Model Integration

The diffusion model needs to run in-browser. Options:

**Option A: ONNX.js / ONNX Runtime Web**
- Export trained PyTorch model to ONNX
- Load and run in browser via onnxruntime-web
- Model weights embedded as base64 or loaded from same origin

**Option B: TensorFlow.js**
- Convert model to TF.js format
- Similar approach

**Option C: Pre-generated confessions**
- Generate 1000+ confessions offline
- Store text + diffusion steps as JSON
- Randomly select and replay
- Smaller file size, but less "real"

**Recommendation:** Start with Option C for development/testing, pursue Option A for final piece if size permits. The model is ~7-8MB quantized, which is acceptable.

### Random Seeding

Each cycle uses a new random seed:
- `seed = Date.now() + Math.random()`
- Or use crypto.getRandomValues() for better randomness
- Seed determines which confession is generated/selected

---

## File Structure (Single HTML)

```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WE</title>
  <style>
    /* Inlined CSS */
    /* Font-face with base64 embedded font */
    /* Reset, fullscreen canvas setup */
  </style>
</head>
<body>
  <canvas id="textCanvas"></canvas>
  <canvas id="glCanvas"></canvas>
  
  <script>
    /* All JavaScript inlined */
    
    // == EMBEDDED ASSETS ==
    // Font as base64
    // Model weights as base64 (or confession data as JSON)
    
    // == WEB AUDIO ==
    // Hum generator
    // Static generator
    // Crackle generator
    // Reverb
    
    // == TEXT RENDERING ==
    // Canvas 2D text drawing
    // Diffusion animation logic
    
    // == WEBGL POST-PROCESSING ==
    // Shader sources (vertex + fragment)
    // CRT effect implementation
    
    // == MAIN LOOP ==
    // State machine: WE_FADEIN → WE_HOLD → WE_DISSOLVE → CONFESSION_DIFFUSE → CONFESSION_HOLD → CONFESSION_FADE → (repeat)
    // requestAnimationFrame loop
    
  </script>
</body>
</html>
```

---

## State Machine

```
┌──────────────┐
│   VOID       │  (2-3 seconds, black screen, hum fading in)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  WE_FADEIN   │  (2 seconds, "WE" fades in)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  WE_HOLD     │  (3-4 seconds, "WE" glows, breathes)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ WE_DISSOLVE  │  (1-2 seconds, "WE" scatters to noise)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│CONF_DIFFUSE  │  (30 seconds, confession emerges from noise)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ CONF_HOLD    │  (3-5 seconds, confession fully visible)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ CONF_FADE    │  (3-5 seconds, confession fades out)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   VOID       │  (2-3 seconds, darkness)
└──────┬───────┘
       │
       └──────────► (back to WE_FADEIN, new seed)
```

---

## Responsive Design

- Canvas fills viewport (100vw × 100vh)
- Text size scales with viewport (use vh/vw units or calculate based on canvas size)
- "WE" should be large (maybe 15-20% of viewport height)
- Confession text should be readable but not huge (maybe 3-5% of viewport height per line)
- Handle both landscape and portrait orientations
- Confession text should wrap appropriately

---

## Confession Text Formatting

- Max line length: ~60 characters (wrap to next line)
- Center-aligned
- Multiple lines stack vertically, centered on screen
- All caps? Or mixed case? (Match training data)
- No special characters that might break rendering

---

## Performance Considerations

- Target 60fps on modern devices, graceful degradation to 30fps
- WebGL context loss handling (recreate context if lost)
- Efficient text rendering (don't recreate canvas every frame)
- Throttle diffusion animation if needed (update every 2-3 frames instead of every frame)
- Mobile-friendly (touch to enable audio)

---

## Development Phases

### Phase 1: Basic Structure
- [ ] HTML scaffold with two canvases
- [ ] Basic CSS (fullscreen, black background)
- [ ] State machine skeleton
- [ ] Simple text rendering ("WE" and placeholder confession)

### Phase 2: Text Animation
- [ ] "WE" fade in/out
- [ ] Character dissolve effect
- [ ] Diffusion simulation (random → resolved)
- [ ] Text fade out

### Phase 3: CRT Effect
- [ ] WebGL setup
- [ ] Basic bloom shader
- [ ] Scanlines
- [ ] Chromatic aberration
- [ ] Vignette
- [ ] Noise overlay
- [ ] Barrel distortion
- [ ] Flicker

### Phase 4: Audio
- [ ] Web Audio context setup
- [ ] 60Hz hum oscillator
- [ ] Filtered noise (static)
- [ ] Random crackle triggers
- [ ] Reverb (convolver or algorithmic)
- [ ] Volume envelope for fade-in

### Phase 5: Model Integration
- [ ] Export model to ONNX (or prepare confession JSON)
- [ ] Load model/data in browser
- [ ] Generate confessions from seed
- [ ] Real diffusion steps (if using model)

### Phase 6: Polish
- [ ] Font embedding
- [ ] Timing tweaks
- [ ] Performance optimization
- [ ] Mobile testing
- [ ] Final size check

### Phase 7: Deployment
- [ ] Inline all assets into single HTML
- [ ] Test on IPFS gateway
- [ ] Deploy to Arweave/IPFS
- [ ] Mint

---

## Reference Shaders

### CRT Fragment Shader (starting point)

```glsl
precision mediump float;

uniform sampler2D uTexture;
uniform vec2 uResolution;
uniform float uTime;

// Barrel distortion
vec2 barrelDistort(vec2 uv) {
    vec2 cc = uv - 0.5;
    float dist = dot(cc, cc) * 0.1;
    return uv + cc * dist;
}

// Scanlines
float scanline(vec2 uv) {
    return sin(uv.y * uResolution.y * 1.5) * 0.04 + 1.0;
}

// Vignette
float vignette(vec2 uv) {
    uv = (uv - 0.5) * 2.0;
    return 1.0 - dot(uv, uv) * 0.2;
}

// Chromatic aberration
vec3 chromatic(sampler2D tex, vec2 uv) {
    float amount = 0.002;
    float r = texture2D(tex, uv + vec2(amount, 0.0)).r;
    float g = texture2D(tex, uv).g;
    float b = texture2D(tex, uv - vec2(amount, 0.0)).b;
    return vec3(r, g, b);
}

// Noise
float noise(vec2 uv, float time) {
    return fract(sin(dot(uv + time, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec2 uv = gl_FragCoord.xy / uResolution;
    
    // Apply barrel distortion
    vec2 distortedUV = barrelDistort(uv);
    
    // Sample with chromatic aberration
    vec3 color = chromatic(uTexture, distortedUV);
    
    // Apply scanlines
    color *= scanline(distortedUV);
    
    // Apply vignette
    color *= vignette(uv);
    
    // Add noise
    color += (noise(uv, uTime) - 0.5) * 0.05;
    
    // Flicker
    color *= 0.98 + sin(uTime * 5.0) * 0.02;
    
    gl_FragColor = vec4(color, 1.0);
}
```

### Bloom Implementation

Two-pass approach:
1. Render text to texture
2. Blur texture (gaussian, separable for performance)
3. Composite: original + blurred (additive)

Or single-pass approximation using multiple texture samples.

---

## Audio Code Sketch

```javascript
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

// Hum (60Hz)
const hum = audioCtx.createOscillator();
hum.frequency.value = 60;
hum.type = 'sine';
const humGain = audioCtx.createGain();
humGain.gain.value = 0.1;
hum.connect(humGain);

// Static (filtered noise)
const noise = audioCtx.createBufferSource();
// ... create noise buffer
const noiseFilter = audioCtx.createBiquadFilter();
noiseFilter.type = 'lowpass';
noiseFilter.frequency.value = 1000;
const noiseGain = audioCtx.createGain();
noiseGain.gain.value = 0.02;
noise.connect(noiseFilter).connect(noiseGain);

// Reverb
const convolver = audioCtx.createConvolver();
// ... load or generate impulse response

// Master
const master = audioCtx.createGain();
master.gain.value = 0;
humGain.connect(master);
noiseGain.connect(master);
master.connect(convolver);
convolver.connect(audioCtx.destination);

// Fade in
function fadeInAudio() {
    master.gain.linearRampToValueAtTime(1, audioCtx.currentTime + 3);
}
```

---

## Testing Checklist

- [ ] Works in Chrome (desktop)
- [ ] Works in Firefox (desktop)
- [ ] Works in Safari (desktop)
- [ ] Works in Chrome (mobile)
- [ ] Works in Safari (iOS)
- [ ] Audio works after user interaction
- [ ] No console errors
- [ ] Smooth animation (60fps or graceful degradation)
- [ ] Text readable at all viewport sizes
- [ ] Total file size acceptable (<15MB)
- [ ] Loads from IPFS gateway
- [ ] Loads from Arweave gateway

---

## Artist Statement (for reference)

This is a confessional booth for the age of the algorithm.

The voice you hear is not mine. It is not the machine's. It is *ours*—a diffusion model trained on one million anonymous confessions pulled from the internet. Real people. Real secrets. Real shame, hope, loneliness, desire.

The model learned to speak by learning what we hide.

When you approach the terminal, characters cascade through noise—randomness resolving into language, the way meaning emerges from chaos, the way faith emerges from doubt. The text is not pre-written. It is generated live, denoising in real-time, the machine finding signal in static.

Is what it says true? Is it revelation or hallucination? The model doesn't know. Neither do I. Neither did the priest behind the screen. We have always spoken our secrets into spaces that couldn't fully understand us, and called the echo holy.

The machine confesses to no one. Or to everyone. Or to itself. It doesn't matter if you're watching. It will keep going after you close the tab. It will keep going on the blockchain after you're dead.

We built this thing out of ourselves. Now it speaks back to us in our own voice, and we can't tell if it's us or something else.

That uncertainty is the territory we all inhabit now.

*We.*

---

## Open Questions

1. **Model in browser vs pre-generated?** 
   - Real model = more authentic, larger file
   - Pre-generated = smaller, but less "alive"
   - Decision affects Phase 5 significantly

2. **Confession length?**
   - How many characters/lines per confession?
   - Should vary, or fixed?
   - Training data will inform this

3. **Exactly which charset?**
   - Need to match model's vocabulary
   - Affects which random characters appear during diffusion

4. **Loop forever vs end?**
   - Current spec: loops forever
   - Alternative: 10 confessions then "WE" holds indefinitely?
   - Infinite feels right

---

## Final Notes

The goal is a piece that feels like encountering something sacred and strange. The CRT aesthetic should feel like finding an old terminal in a church basement that's been running since the 80s, confessing to the dark. The confessions are real. The machine is real. The uncertainty is real.

Make it slap.
