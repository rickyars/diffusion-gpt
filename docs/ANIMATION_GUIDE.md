# Creating Denoising Animations

Generate cool animated GIFs showing text emerging from random noise, just like the demo in the original repo!

## Quick Start

```bash
# Install Pillow if not already installed
pip install Pillow

# Generate an animation from your trained model
python generate_animation.py --model models/shakespeare.pt --output outputs/my_animation.gif
```

That's it! You'll get an animated GIF showing the complete denoising process.

## Example Output

The animation shows:
- **Step 0**: Random noise (completely garbled characters)
- **Steps 1-127**: Gradual denoising (text becomes more coherent)
- **Step 128**: Final clean text (Shakespeare-style output)

Each frame displays:
- Current step number
- Noise level (σ)
- The denoised text at that step

## Usage

### Basic Usage

```bash
python generate_animation.py --model models/shakespeare.pt
```

This uses default settings:
- 128 denoising steps
- 800x600 resolution
- 200ms per frame
- Saves all frames

### Custom Settings

```bash
python generate_animation.py \
  --model models/shakespeare.pt \
  --output outputs/demo.gif \
  --steps 64 \
  --width 1000 \
  --height 800 \
  --duration 150 \
  --save-every 2 \
  --seed 42
```

### All Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | (required) | Path to trained model (.pt file) |
| `--vocab` | Auto-detect | Path to vocabulary .pkl file |
| `--output` | outputs/denoising_animation.gif | Output path |
| `--steps` | 128 | Number of denoising steps |
| `--width` | 800 | Image width in pixels |
| `--height` | 600 | Image height in pixels |
| `--duration` | 200 | Frame duration in milliseconds |
| `--save-every` | 1 | Save every Nth frame (1=all frames) |
| `--seed` | 42 | Random seed for reproducibility |
| `--device` | auto | cpu or cuda |
| `--config` | config.yaml | Path to config file |

## Tips for Great Animations

### Faster Generation (Fewer Frames)

If you want a smaller file and faster generation:

```bash
python generate_animation.py \
  --model models/shakespeare.pt \
  --save-every 4 \
  --steps 64
```

This saves every 4th frame from 64 steps = only 16 frames (~1MB instead of ~10MB).

### Slower Animation (More Readable)

To make the animation slower so viewers can read the text:

```bash
python generate_animation.py \
  --model models/shakespeare.pt \
  --duration 300
```

300ms per frame = slower, more readable animation.

### High Quality for Presentations

For high-resolution animations:

```bash
python generate_animation.py \
  --model models/shakespeare.pt \
  --width 1200 \
  --height 900 \
  --steps 128 \
  --duration 100
```

### Quick Demo (Small File)

For a quick demo with a small file size:

```bash
python generate_animation.py \
  --model models/shakespeare.pt \
  --steps 32 \
  --save-every 2 \
  --width 600 \
  --height 400
```

This creates a ~1-2MB file with 16 frames.

## Examples

### Example 1: Shakespeare Model

```bash
python generate_animation.py \
  --model models/shakespeare.pt \
  --output outputs/shakespeare_denoising.gif
```

### Example 2: Custom Dataset

```bash
python generate_animation.py \
  --model models/my_dataset.pt \
  --vocab vocab/my_dataset_vocab.pkl \
  --output outputs/my_animation.gif \
  --steps 64
```

### Example 3: Fast Preview

```bash
python generate_animation.py \
  --model models/shakespeare.pt \
  --steps 32 \
  --save-every 2 \
  --output outputs/preview.gif
```

## Understanding the Output

### File Size

File size depends on:
- **Number of frames**: More frames = larger file
- **Resolution**: Larger images = larger file
- **Text complexity**: More varied text = less compression

Typical sizes:
- 64 steps, all frames, 800x600: ~5-10 MB
- 128 steps, all frames, 800x600: ~10-20 MB
- 32 steps, every 2nd frame, 600x400: ~1-2 MB

### Frame Count

The script saves:
- 1 initial frame (random noise)
- 1 frame every N steps (based on `--save-every`)
- 5 duplicate final frames (so viewers can read the result)

Formula: `frames = (steps / save_every) + 6`

### Animation Duration

Total animation time = `frames × duration / 1000` seconds

Examples:
- 128 frames × 200ms = 25.6 seconds
- 64 frames × 150ms = 9.6 seconds
- 32 frames × 300ms = 9.6 seconds

## Technical Details

### How It Works

1. **Start with noise**: Random character sequence (length = model's block_size)
2. **Iterative denoising**: Run the diffusion model for N steps
3. **Capture frames**: Convert text to image at each step (or every Nth step)
4. **Create GIF**: Combine all frames into an animated GIF

### Text-to-Image Rendering

The script uses PIL (Pillow) to:
- Render text with a monospace font
- Add step information header
- Use dark theme for better visibility
- Wrap text to fit the image width

### Font Selection

The script tries to use system fonts in this order:
1. DejaVu Sans Mono (Linux)
2. Courier (macOS)
3. Courier New (Windows)
4. Liberation Mono (Linux alternative)
5. Default bitmap font (fallback)

## Troubleshooting

### "No module named 'PIL'"

Install Pillow:
```bash
pip install Pillow
```

### "Vocabulary file not found"

Specify the vocab file explicitly:
```bash
python generate_animation.py \
  --model models/shakespeare.pt \
  --vocab vocab/shakespeare_vocab.pkl
```

### Animation is too fast

Increase frame duration:
```bash
python generate_animation.py --model models/shakespeare.pt --duration 300
```

### File size too large

Reduce frames or resolution:
```bash
python generate_animation.py \
  --model models/shakespeare.pt \
  --save-every 4 \
  --width 600 \
  --height 400
```

### Text is cut off

Increase image height:
```bash
python generate_animation.py --model models/shakespeare.pt --height 800
```

## Sharing Your Animations

Great places to share:
- GitHub repo (add to README.md)
- Twitter/X
- LinkedIn
- Research papers
- Presentations

GitHub Markdown:
```markdown
![Denoising Process](outputs/my_animation.gif)
```

## Advanced: Creating Custom Visualizations

The `text_to_image()` function can be customized to:
- Change colors
- Add custom headers/footers
- Highlight specific characters
- Use different fonts
- Add background patterns

Check the source code in `generate_animation.py` for customization options.

## Comparison with Original Repo

This implementation creates animations similar to the original discrete diffusion paper demos, showing:
- Initial random noise
- Gradual refinement over steps
- Final coherent text output
- Step-by-step progress information

The visualization makes it easy to understand how discrete diffusion works!
