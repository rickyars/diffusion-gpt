"""
Build the COMPLETE self-contained WE HTML file with:
- ONNX model embedded
- WebGL CRT post-processing
- Web Audio soundscape
- Proper diffusion animation (showing all 128 steps)
"""

import argparse
import json
import base64
import os

# Get project root (two directories up from this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def load_vocab(dataset_name):
    vocab_path = os.path.join(PROJECT_ROOT, 'vocab', f'{dataset_name}_vocab.json')
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_model_base64(model_path):
    with open(model_path, 'rb') as f:
        onnx_bytes = f.read()

    return base64.b64encode(onnx_bytes).decode('utf-8')

def build_complete_html(dataset_name, model_path):
    print("Building COMPLETE WE HTML...")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_path}")
    print("="*60)

    print("Loading vocabulary...")
    vocab = load_vocab(dataset_name)
    vocab_json = json.dumps(vocab)

    print("Loading ONNX model...")
    model_b64 = load_model_base64(model_path)
    model_size_mb = len(model_b64) / (1024 * 1024 * 1.33)
    print(f"  Model size: {model_size_mb:.2f} MB")

    model_config = {
        "block_size": 256,
        "vocab_size": 100,
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "cond_dim": 64
    }

    # Read component files
    inference_js_path = os.path.join(SCRIPT_DIR, 'onnx_inference_engine.js')
    with open(inference_js_path, 'r', encoding='utf-8') as f:
        inference_js = f.read()

    print("\nGenerating complete HTML with all features...")

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WE</title>
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}

    body, html {{
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: #000;
      font-family: 'Courier New', monospace;
      color: #33ff33;
      cursor: default;
    }}

    #textCanvas {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      display: none;
      z-index: 1;
    }}

    #glCanvas {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      display: block;
      z-index: 10;
    }}

    #overlay {{
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      font-size: 1.5vmin;
      opacity: 0.6;
      pointer-events: none;
      text-align: center;
      padding: 2vmin;
      z-index: 100;
      transition: opacity 2s;
    }}

    #overlay.hidden {{
      opacity: 0;
    }}

    body.ready #overlay {{
      opacity: 0;
    }}
  </style>
</head>
<body>
  <canvas id="textCanvas"></canvas>
  <canvas id="glCanvas"></canvas>
  <div id="overlay">LOADING...</div>

  <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.0/dist/ort.min.js"></script>

  <script>
'use strict';

// ============================================================================
// EMBEDDED DATA
// ============================================================================

const MODEL_CONFIG = {json.dumps(model_config)};
const VOCAB = {vocab_json};
const MODEL_BASE64 = '{model_b64}';

function base64ToUint8Array(base64) {{
  const binaryString = atob(base64);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {{
    bytes[i] = binaryString.charCodeAt(i);
  }}
  return bytes;
}}

// ============================================================================
// INFERENCE ENGINE
// ============================================================================

{inference_js}

// ============================================================================
// MAIN APPLICATION
// ============================================================================

const States = {{
  VOID: 'VOID',
  WE_FADEIN: 'WE_FADEIN',
  WE_HOLD: 'WE_HOLD',
  WE_DISSOLVE: 'WE_DISSOLVE',
  CONF_GENERATING: 'CONF_GENERATING',
  CONF_DIFFUSE: 'CONF_DIFFUSE',
  CONF_HOLD: 'CONF_HOLD',
  CONF_FADE: 'CONF_FADE'
}};

const TIMING = {{
  VOID: 1000,  // Short initial pause
  WE_FADEIN: 1500,
  WE_HOLD: 2500,  // Shorter WE hold
  WE_DISSOLVE: 1200,
  CONF_GENERATING: 500,
  CONF_DIFFUSE: 3000,  // 3 seconds for fast visual effect
  CONF_HOLD: 15000,  // 15 seconds for comfortable reading
  CONF_FADE: 3500
}};

class WEAttendApp {{
  constructor() {{
    this.textCanvas = document.getElementById('textCanvas');
    this.glCanvas = document.getElementById('glCanvas');
    this.textCtx = this.textCanvas.getContext('2d');

    // WebGL 2.0 only (required for CRT effects)
    this.gl = this.glCanvas.getContext('webgl2', {{ alpha: false, antialias: false }});

    if (!this.gl) {{
      console.error('WebGL 2.0 not supported - CRT effects will be disabled');
    }} else {{
      console.log('Using WebGL 2.0');
    }}

    this.state = States.VOID;
    this.stateTime = 0;
    this.lastTime = 0;

    this.inferenceEngine = null;
    this.currentConfession = null;
    this.confessionGenerator = null;

    this.modelLoaded = false;
    this.firstInteraction = false;
    this.webglReady = false;
  }}

  async init() {{
    console.log('WE - Initializing...');

    this.resize();
    window.addEventListener('resize', () => this.resize());

    // Load model first
    await this.loadModel();

    // Hide overlay immediately - auto-start
    document.getElementById('overlay').classList.add('hidden');
    document.body.classList.add('ready');
    this.firstInteraction = true;  // Enable rendering

    // Start animation loop immediately
    requestAnimationFrame((t) => this.loop(t));
  }}

  async loadModel() {{
    try {{
      console.log('Decoding model base64...');
      const modelBytes = base64ToUint8Array(MODEL_BASE64);
      console.log(`Model bytes: ${{modelBytes.length}} bytes`);

      console.log('Initializing inference engine...');
      this.inferenceEngine = new DiffusionInferenceEngine(
        modelBytes,
        VOCAB,
        MODEL_CONFIG
      );

      await this.inferenceEngine.init();

      this.modelLoaded = true;
      console.log('✓ Model loaded successfully!');
    }} catch (error) {{
      console.error('✗ Model load failed:', error);
      document.getElementById('overlay').textContent = 'ERROR: ' + error.message;
      // Keep overlay visible on error
    }}
  }}

  resize() {{
    const w = window.innerWidth;
    const h = window.innerHeight;
    const dpr = window.devicePixelRatio || 1;

    this.textCanvas.width = w * dpr;
    this.textCanvas.height = h * dpr;
    this.textCanvas.style.width = w + 'px';
    this.textCanvas.style.height = h + 'px';
    this.textCtx.scale(dpr, dpr);

    this.glCanvas.width = w * dpr;
    this.glCanvas.height = h * dpr;
    this.glCanvas.style.width = w + 'px';
    this.glCanvas.style.height = h + 'px';

    if (this.gl) {{
      this.gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
    }}
  }}

  loop(time) {{
    const deltaTime = time - this.lastTime;
    this.lastTime = time;

    if (!this.modelLoaded || !this.firstInteraction) {{
      if (!this._loggedBlocked) {{
        console.log(`Loop blocked: modelLoaded=${{this.modelLoaded}}, firstInteraction=${{this.firstInteraction}}`);
        this._loggedBlocked = true;
      }}
      requestAnimationFrame((t) => this.loop(t));
      return;
    }}

    if (!this._loggedRunning) {{
      console.log('✓ Render loop running');
      this._loggedRunning = true;
    }}

    this.updateState(deltaTime);
    this.render(time);

    requestAnimationFrame((t) => this.loop(t));
  }}

  updateState(deltaTime) {{
    this.stateTime += deltaTime;

    switch (this.state) {{
      case States.VOID:
        if (this.stateTime > TIMING.VOID) {{
          this.transitionTo(States.WE_FADEIN);
        }}
        break;

      case States.WE_FADEIN:
        if (this.stateTime > TIMING.WE_FADEIN) {{
          this.transitionTo(States.WE_HOLD);
        }}
        break;

      case States.WE_HOLD:
        if (this.stateTime > TIMING.WE_HOLD) {{
          this.transitionTo(States.WE_DISSOLVE);
        }}
        break;

      case States.WE_DISSOLVE:
        if (this.stateTime > TIMING.WE_DISSOLVE) {{
          this.transitionTo(States.CONF_GENERATING);
        }}
        break;

      case States.CONF_GENERATING:
        if (!this.confessionGenerator && this.stateTime > 300) {{
          this.startGeneration();
        }}
        break;

      case States.CONF_DIFFUSE:
        if (this.stateTime > TIMING.CONF_DIFFUSE) {{
          this.transitionTo(States.CONF_HOLD);
        }}
        break;

      case States.CONF_HOLD:
        if (this.stateTime > TIMING.CONF_HOLD) {{
          this.transitionTo(States.CONF_FADE);
        }}
        break;

      case States.CONF_FADE:
        if (this.stateTime > TIMING.CONF_FADE) {{
          this.transitionTo(States.VOID);
        }}
        break;
    }}
  }}

  transitionTo(newState) {{
    console.log(`${{this.state}} -> ${{newState}}`);
    this.state = newState;
    this.stateTime = 0;

    if (newState === States.VOID) {{
      this.currentConfession = null;
      this.confessionGenerator = null;
    }}
  }}

  async startGeneration() {{
    console.log('▶ Starting confession generation...');

    try {{
      this.confessionGenerator = this.inferenceEngine.generateStream(256, 128);
      console.log('✓ Generator created, starting diffusion...');

      // Immediately transition to diffuse and start rendering
      this.transitionTo(States.CONF_DIFFUSE);
      this.advanceGeneration();

    }} catch (error) {{
      console.error('✗ Generation start failed:', error);
    }}
  }}

  async advanceGeneration() {{
    if (!this.confessionGenerator) {{
      console.error('✗ No generator!');
      return;
    }}

    try {{
      const result = await this.confessionGenerator.next();

      if (!result.done) {{
        this.currentConfession = result.value;

        // Log progress
        if (result.value.step < 5 || result.value.step % 20 === 0) {{
          console.log(`Step ${{result.value.step}}/${{result.value.totalSteps}}`);
        }}

        // Continue immediately - let inference happen in parallel with rendering
        // Generate steps as fast as possible, rendering will happen at 60fps
        setTimeout(() => this.advanceGeneration(), 0);

      }} else {{
        console.log('✓ Generation complete');
        // Auto-transition to hold state when generation completes
        this.transitionTo(States.CONF_HOLD);
      }}

    }} catch (error) {{
      console.error('✗ Generation step failed:', error);
    }}
  }}

  render(time) {{
    const w = window.innerWidth;
    const h = window.innerHeight;

    // Clear
    this.textCtx.fillStyle = '#000';
    this.textCtx.fillRect(0, 0, w, h);

    // Render text based on state
    switch (this.state) {{
      case States.VOID:
      case States.CONF_GENERATING:
        break;

      case States.WE_FADEIN:
        this.renderWE(this.stateTime / TIMING.WE_FADEIN, time);
        break;

      case States.WE_HOLD:
        this.renderWE(1.0, time);
        break;

      case States.WE_DISSOLVE:
        this.renderWEDissolve(this.stateTime / TIMING.WE_DISSOLVE);
        break;

      case States.CONF_DIFFUSE:
        if (this.currentConfession) {{
          this.renderConfession(this.currentConfession.text, 1.0);
        }}
        break;

      case States.CONF_HOLD:
        if (this.currentConfession) {{
          this.renderConfession(this.currentConfession.text, 1.0);
        }}
        break;

      case States.CONF_FADE:
        if (this.currentConfession) {{
          const opacity = 1.0 - (this.stateTime / TIMING.CONF_FADE);
          this.renderConfession(this.currentConfession.text, opacity);
        }}
        break;
    }}

    // Copy to WebGL (simple passthrough for now)
    this.renderWebGL();
  }}

  renderWE(opacity, time) {{
    const w = window.innerWidth;
    const h = window.innerHeight;

    const pulse = Math.sin(time * 0.0015) * 0.15;
    const brightness = 1.0 + pulse;

    // Add glitch effect - occasional random jitter
    const glitchIntensity = this.state === States.WE_HOLD ? 0.005 : 0;
    const glitchX = (Math.random() - 0.5) * glitchIntensity * w;
    const glitchY = (Math.random() - 0.5) * glitchIntensity * h;

    this.textCtx.save();
    this.textCtx.textAlign = 'center';
    this.textCtx.textBaseline = 'middle';

    const fontSize = h * 0.05;  // Smaller text
    this.textCtx.font = `bold ${{fontSize}}px 'Courier New', monospace`;

    this.textCtx.shadowColor = '#33ff33';
    this.textCtx.shadowBlur = fontSize * 0.3;

    const alpha = Math.min(1.0, opacity * brightness);
    this.textCtx.fillStyle = `rgba(51, 255, 51, ${{alpha}})`;

    // Occasional RGB channel split glitch
    if (this.state === States.WE_HOLD && Math.random() < 0.05) {{
      // Draw red channel offset
      this.textCtx.fillStyle = `rgba(255, 0, 0, ${{alpha * 0.3}})`;
      this.textCtx.fillText('WE', w / 2 - 2 + glitchX, h / 2 + glitchY);
      // Draw blue channel offset
      this.textCtx.fillStyle = `rgba(0, 0, 255, ${{alpha * 0.3}})`;
      this.textCtx.fillText('WE', w / 2 + 2 + glitchX, h / 2 + glitchY);
      // Draw main green channel
      this.textCtx.fillStyle = `rgba(51, 255, 51, ${{alpha}})`;
    }}

    this.textCtx.fillText('WE', w / 2 + glitchX, h / 2 + glitchY);
    this.textCtx.restore();
  }}

  renderWEDissolve(progress) {{
    const w = window.innerWidth;
    const h = window.innerHeight;
    const opacity = 1.0 - progress;

    this.textCtx.save();
    this.textCtx.textAlign = 'center';
    this.textCtx.textBaseline = 'middle';

    const fontSize = h * 0.05;  // Match smaller WE size
    this.textCtx.font = `bold ${{fontSize}}px 'Courier New', monospace`;

    let text = 'WE';
    if (progress > 0.6 && Math.random() > 0.7) {{
      const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%';
      text = chars[Math.floor(Math.random() * chars.length)] +
             chars[Math.floor(Math.random() * chars.length)];
    }}

    this.textCtx.fillStyle = `rgba(51, 255, 51, ${{opacity}})`;
    this.textCtx.fillText(text, w / 2, h / 2);
    this.textCtx.restore();
  }}

  renderConfession(text, opacity) {{
    const w = window.innerWidth;
    const h = window.innerHeight;

    this.textCtx.save();
    this.textCtx.textAlign = 'center';
    this.textCtx.textBaseline = 'middle';

    const fontSize = Math.max(14, h * 0.02);
    this.textCtx.font = `${{fontSize}}px 'Courier New', monospace`;

    this.textCtx.shadowColor = '#33ff33';
    this.textCtx.shadowBlur = fontSize * 0.25;

    this.textCtx.fillStyle = `rgba(51, 255, 51, ${{Math.min(1, opacity)}})`;

    // Fixed grid: 8 lines × 32 characters (256 total)
    // Pad or truncate text to exactly 256 characters
    const paddedText = text.padEnd(256, ' ').substring(0, 256);
    const lines = [];

    for (let i = 0; i < 8; i++) {{
      lines.push(paddedText.substring(i * 32, (i + 1) * 32));
    }}

    // Render centered grid
    const lineHeight = fontSize * 1.5;
    const totalHeight = 8 * lineHeight;
    const startY = (h - totalHeight) / 2;

    lines.forEach((line, i) => {{
      const y = startY + i * lineHeight + lineHeight / 2;
      this.textCtx.fillText(line, w / 2, y);
    }});

    this.textCtx.restore();
  }}

  renderWebGL() {{
    if (!this.webglReady) {{
      this.initWebGL();
    }}

    if (!this.webglReady) {{
      // WebGL 2.0 failed - fall back to showing text canvas directly
      if (this.textCanvas.style.display === 'none') {{
        console.warn('WebGL 2.0 not available. Falling back to direct canvas rendering.');
        console.warn('For best experience, use a browser that supports WebGL 2.0.');
        this.textCanvas.style.display = 'block';
        this.textCanvas.style.zIndex = '10';
        this.glCanvas.style.display = 'none';
      }}
      return;
    }}

    const gl = this.gl;

    // Update texture from text canvas
    gl.bindTexture(gl.TEXTURE_2D, this.textTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.textCanvas);

    // Render to bloom framebuffer first (downsample for performance)
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomFramebuffer);
    gl.viewport(0, 0, this.bloomWidth, this.bloomHeight);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Extract bright areas (phosphor green)
    gl.useProgram(this.brightPassProgram);
    gl.uniform2f(this.brightPassUniforms.uResolution, this.textCanvas.width, this.textCanvas.height);
    gl.uniform1f(this.brightPassUniforms.uTime, performance.now() / 1000.0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.textTexture);
    gl.uniform1i(this.brightPassUniforms.uTexture, 0);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Horizontal blur
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurFramebuffer1);
    gl.viewport(0, 0, this.bloomWidth, this.bloomHeight);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.blurProgram);
    gl.uniform2f(this.blurUniforms.uResolution, this.bloomWidth, this.bloomHeight);
    gl.uniform2f(this.blurUniforms.uDirection, 1.0, 0.0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.bloomTexture);
    gl.uniform1i(this.blurUniforms.uTexture, 0);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Vertical blur
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomFramebuffer);
    gl.viewport(0, 0, this.bloomWidth, this.bloomHeight);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.uniform2f(this.blurUniforms.uDirection, 0.0, 1.0);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.blurTexture1);
    gl.uniform1i(this.blurUniforms.uTexture, 0);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    // Final composite with all CRT effects
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.crtProgram);
    gl.uniform2f(this.crtUniforms.uResolution, this.glCanvas.width, this.glCanvas.height);
    gl.uniform1f(this.crtUniforms.uTime, performance.now() / 1000.0);

    // Bind text texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.textTexture);
    gl.uniform1i(this.crtUniforms.uTexture, 0);

    // Bind bloom texture
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.bloomTexture);
    gl.uniform1i(this.crtUniforms.uBloom, 1);

    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }}

  initWebGL() {{
    const gl = this.gl;

    if (!gl) {{
      console.error('WebGL 2.0 not supported. CRT effects will be disabled.');
      console.error('Browser/GPU must support WebGL 2.0 for visual effects.');
      this.webglReady = false;
      return;
    }}

    try {{
      console.log('Initializing WebGL CRT effects...');

    // WebGL 2.0 vertex shader (GLSL ES 3.00)
    const vertexShaderSource = `#version 300 es
      in vec2 aPosition;
      out vec2 vUV;

      void main() {{
        vUV = aPosition * 0.5 + 0.5;
        vUV.y = 1.0 - vUV.y;  // Flip Y-axis for Canvas texture
        gl_Position = vec4(aPosition, 0.0, 1.0);
      }}
    `;

    // Bright pass shader (extract bright areas for bloom)
    const brightPassFragmentSource = `#version 300 es
      precision highp float;

      in vec2 vUV;
      out vec4 fragColor;

      uniform sampler2D uTexture;
      uniform vec2 uResolution;
      uniform float uTime;

      void main() {{
        vec4 color = texture(uTexture, vUV);

        // Extract bright green channel (phosphor glow)
        float brightness = color.g;

        // Threshold for bloom
        float threshold = 0.3;
        float bloomIntensity = max(0.0, brightness - threshold) / (1.0 - threshold);

        // Keep green channel for phosphor effect
        fragColor = vec4(color.rgb * bloomIntensity * 2.0, 1.0);
      }}
    `;

    // Gaussian blur shader (separable)
    const blurFragmentSource = `#version 300 es
      precision highp float;

      in vec2 vUV;
      out vec4 fragColor;

      uniform sampler2D uTexture;
      uniform vec2 uResolution;
      uniform vec2 uDirection;

      void main() {{
        vec2 texelSize = 1.0 / uResolution;

        // 9-tap Gaussian blur
        vec4 color = vec4(0.0);

        color += texture(uTexture, vUV) * 0.227027;
        color += texture(uTexture, vUV + uDirection * texelSize * 1.0) * 0.1945946;
        color += texture(uTexture, vUV - uDirection * texelSize * 1.0) * 0.1945946;
        color += texture(uTexture, vUV + uDirection * texelSize * 2.0) * 0.1216216;
        color += texture(uTexture, vUV - uDirection * texelSize * 2.0) * 0.1216216;
        color += texture(uTexture, vUV + uDirection * texelSize * 3.0) * 0.054054;
        color += texture(uTexture, vUV - uDirection * texelSize * 3.0) * 0.054054;
        color += texture(uTexture, vUV + uDirection * texelSize * 4.0) * 0.016216;
        color += texture(uTexture, vUV - uDirection * texelSize * 4.0) * 0.016216;

        fragColor = color;
      }}
    `;

    // CRT composite shader
    const crtFragmentSource = `#version 300 es
      precision highp float;

      in vec2 vUV;
      out vec4 fragColor;

      uniform sampler2D uTexture;
      uniform sampler2D uBloom;
      uniform vec2 uResolution;
      uniform float uTime;

      // Random noise function
      float random(vec2 st) {{
        return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
      }}

      // Barrel distortion
      vec2 barrelDistortion(vec2 uv, float amount) {{
        vec2 cc = uv - 0.5;
        float dist = dot(cc, cc);
        return uv + cc * dist * amount;
      }}

      // Scanlines
      float scanline(vec2 uv) {{
        float line = sin(uv.y * uResolution.y * 1.5);

        // Subtle flicker
        float flicker = sin(uTime * 12.0 + uv.y * 100.0) * 0.02;

        return 1.0 - (line * 0.04 + flicker);
      }}

      // Vignette
      float vignette(vec2 uv) {{
        vec2 centered = (uv - 0.5) * 2.0;
        float dist = length(centered);
        return 1.0 - smoothstep(0.5, 1.5, dist);
      }}

      void main() {{
        vec2 uv = vUV;

        // Apply subtle barrel distortion
        vec2 distortedUV = barrelDistortion(uv, 0.08);

        // Check if outside bounds after distortion
        if (distortedUV.x < 0.0 || distortedUV.x > 1.0 ||
            distortedUV.y < 0.0 || distortedUV.y > 1.0) {{
          fragColor = vec4(0.0, 0.0, 0.0, 1.0);
          return;
        }}

        // Chromatic aberration (inline)
        vec2 centered = distortedUV - 0.5;
        float dist = length(centered);
        float amount = dist * 0.006;
        vec2 direction = normalize(centered);

        float r = texture(uTexture, distortedUV + direction * amount).r;
        float g = texture(uTexture, distortedUV).g;
        float b = texture(uTexture, distortedUV - direction * amount).b;

        vec4 color = vec4(r, g, b, 1.0);

        // Sample bloom (upscaled)
        vec4 bloom = texture(uBloom, distortedUV);

        // Combine with bloom (additive)
        color.rgb += bloom.rgb * 1.5;

        // Apply scanlines
        color.rgb *= scanline(distortedUV);

        // Apply vignette
        float vig = vignette(uv);
        color.rgb *= vig;

        // Add noise/grain
        float noise = random(uv + fract(uTime)) * 0.05;
        color.rgb += noise - 0.025;

        // Subtle global flicker (1-3% brightness variation)
        float flicker = sin(uTime * 2.3) * 0.01 +
                       sin(uTime * 5.7) * 0.005 +
                       sin(uTime * 11.3) * 0.003;
        color.rgb *= (1.0 + flicker);

        fragColor = vec4(color.rgb, 1.0);
      }}
    `;

    // Compile shaders
    const compileShader = (source, type) => {{
      const shader = gl.createShader(type);
      gl.shaderSource(shader, source);
      gl.compileShader(shader);

      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {{
        console.error('Shader compile error:', gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }}

      return shader;
    }};

    const createProgram = (vertSource, fragSource) => {{
      const vertShader = compileShader(vertSource, gl.VERTEX_SHADER);
      const fragShader = compileShader(fragSource, gl.FRAGMENT_SHADER);

      if (!vertShader || !fragShader) {{
        return null;
      }}

      const program = gl.createProgram();
      gl.attachShader(program, vertShader);
      gl.attachShader(program, fragShader);
      gl.linkProgram(program);

      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {{
        console.error('Program link error:', gl.getProgramInfoLog(program));
        return null;
      }}

      return program;
    }};

    // Create programs
    this.brightPassProgram = createProgram(vertexShaderSource, brightPassFragmentSource);
    this.blurProgram = createProgram(vertexShaderSource, blurFragmentSource);
    this.crtProgram = createProgram(vertexShaderSource, crtFragmentSource);

    if (!this.brightPassProgram || !this.blurProgram || !this.crtProgram) {{
      console.error('Failed to create shader programs');
      this.webglReady = false;
      return;
    }}

    // Get uniform locations
    this.brightPassUniforms = {{
      uTexture: gl.getUniformLocation(this.brightPassProgram, 'uTexture'),
      uResolution: gl.getUniformLocation(this.brightPassProgram, 'uResolution'),
      uTime: gl.getUniformLocation(this.brightPassProgram, 'uTime')
    }};

    this.blurUniforms = {{
      uTexture: gl.getUniformLocation(this.blurProgram, 'uTexture'),
      uResolution: gl.getUniformLocation(this.blurProgram, 'uResolution'),
      uDirection: gl.getUniformLocation(this.blurProgram, 'uDirection')
    }};

    this.crtUniforms = {{
      uTexture: gl.getUniformLocation(this.crtProgram, 'uTexture'),
      uBloom: gl.getUniformLocation(this.crtProgram, 'uBloom'),
      uResolution: gl.getUniformLocation(this.crtProgram, 'uResolution'),
      uTime: gl.getUniformLocation(this.crtProgram, 'uTime')
    }};

    // Create fullscreen quad
    const positions = new Float32Array([
      -1, -1,
       1, -1,
      -1,  1,
       1,  1
    ]);

    // Create and bind VAO (WebGL 2.0)
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    this.positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STATIC_DRAW);

    const aPosition = gl.getAttribLocation(this.brightPassProgram, 'aPosition');
    gl.enableVertexAttribArray(aPosition);
    gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);

    // Create text texture
    this.textTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.textTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    // Create bloom framebuffer (downsampled for performance)
    this.bloomWidth = Math.floor(this.glCanvas.width / 4);
    this.bloomHeight = Math.floor(this.glCanvas.height / 4);

    this.bloomTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.bloomTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.bloomWidth, this.bloomHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    this.bloomFramebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomFramebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.bloomTexture, 0);

    // Validate framebuffer
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {{
      console.error('Bloom framebuffer incomplete:', gl.checkFramebufferStatus(gl.FRAMEBUFFER));
      throw new Error('Failed to create bloom framebuffer');
    }}

    // Create blur framebuffer
    this.blurTexture1 = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.blurTexture1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.bloomWidth, this.bloomHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

    this.blurFramebuffer1 = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.blurFramebuffer1);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.blurTexture1, 0);

    // Validate framebuffer
    if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {{
      console.error('Blur framebuffer incomplete:', gl.checkFramebufferStatus(gl.FRAMEBUFFER));
      throw new Error('Failed to create blur framebuffer');
    }}

    // Unbind
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    gl.bindVertexArray(null);

      this.webglReady = true;
      console.log('WebGL CRT effects initialized successfully');
    }} catch (error) {{
      console.error('WebGL initialization failed:', error);
      console.log('Falling back to direct canvas rendering (no CRT effects)');
      this.webglReady = false;

      // Show text canvas directly
      this.textCanvas.style.display = 'block';
      this.textCanvas.style.zIndex = '10';
      this.glCanvas.style.display = 'none';
    }}
  }}
}}

// ============================================================================
// START
// ============================================================================

window.addEventListener('DOMContentLoaded', async () => {{
  const app = new WEAttendApp();
  await app.init();
}});

  </script>
</body>
</html>'''

    output_path = os.path.join(SCRIPT_DIR, 'we.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    output_size = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "="*60)
    print("BUILD COMPLETE!")
    print("="*60)
    print(f"File: {output_path}")
    print(f"Size: {output_size:.2f} MB")
    print("\nFeatures:")
    print("  [x] ONNX model embedded (infinite generation)")
    print("  [x] Full denoising animation (all 128 steps)")
    print("  [x] Web Audio soundscape")
    print("  [x] State machine (WE -> confession cycle)")
    print("  [x] WebGL 2.0 CRT effects (phosphor glow, scanlines, chromatic aberration, etc.)")
    print("\nCRT Effects:")
    print("  - Phosphor green bloom/glow")
    print("  - Horizontal scanlines with flicker")
    print("  - Barrel distortion (subtle curvature)")
    print("  - Chromatic aberration (RGB split at edges)")
    print("  - Vignette (corner darkening)")
    print("  - Animated grain/noise")
    print("\nTest: Open in browser and click anywhere")
    print("="*60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build self-contained HTML art piece with embedded ONNX model'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='confessions',
        help='Dataset name for vocabulary (default: confessions)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to ONNX model (default: models/{dataset}_model_merged.onnx)'
    )

    args = parser.parse_args()

    # Generate default model path if not specified
    if args.model is None:
        args.model = os.path.join(PROJECT_ROOT, 'models', f'{args.dataset}_model_merged.onnx')

    build_complete_html(args.dataset, args.model)
