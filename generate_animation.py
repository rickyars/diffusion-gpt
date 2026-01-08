"""
Generate animated GIF showing the discrete diffusion denoising process.
Creates a visualization similar to the demo in the original repo.
"""

import argparse
import os
import sys
import textwrap
from typing import List, Tuple

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

from model import GPT, GPTConfig
from utils import (
    GeometricNoise,
    decode,
    sample_categorical,
    set_seed,
    staggered_score,
    transition,
)


def text_to_image(
    text: str,
    width: int = 800,
    height: int = 600,
    font_size: int = 14,
    bg_color: Tuple[int, int, int] = (30, 30, 30),
    text_color: Tuple[int, int, int] = (220, 220, 220),
    step_info: str = "",
) -> Image.Image:
    """
    Convert text to an image with monospace font.

    Args:
        text: Text to render
        width: Image width
        height: Image height
        font_size: Font size
        bg_color: Background RGB color
        text_color: Text RGB color
        step_info: Optional step information to display at top

    Returns:
        PIL Image
    """
    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Try to load a monospace font, fall back to default if not available
    try:
        # Try common monospace fonts
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Courier.dfont",
            "C:\\Windows\\Fonts\\cour.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        ]
        font = None
        for font_path in font_paths:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        if font is None:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # Draw step info at the top
    margin_top = 20
    margin_left = 20

    if step_info:
        # Draw header background
        draw.rectangle([(0, 0), (width, 40)], fill=(50, 50, 70))
        draw.text((margin_left, 10), step_info, fill=(150, 200, 255), font=font)
        margin_top = 60

    # Wrap text to fit width
    chars_per_line = (width - 2 * margin_left) // (font_size // 2)
    wrapped_lines = []

    for line in text.split('\n'):
        if line:
            wrapped = textwrap.wrap(line, width=chars_per_line, break_long_words=True)
            wrapped_lines.extend(wrapped if wrapped else [''])
        else:
            wrapped_lines.append('')

    # Draw text
    y = margin_top
    line_height = font_size + 4

    for line in wrapped_lines:
        if y + line_height > height - 20:
            break
        draw.text((margin_left, y), line, fill=text_color, font=font)
        y += line_height

    return img


def generate_animation(
    model: GPT,
    noise: GeometricNoise,
    vocab_size: int,
    itos: dict,
    context_length: int = 256,
    steps: int = 128,
    device: torch.device = torch.device('cpu'),
    eps: float = 1e-5,
    output_path: str = 'denoising_animation.gif',
    frame_duration: int = 200,
    image_width: int = 800,
    image_height: int = 600,
    save_every: int = 1,
) -> str:
    """
    Generate an animated GIF of the denoising process.

    Args:
        model: Trained GPT model
        noise: GeometricNoise instance
        vocab_size: Vocabulary size
        itos: Index to string mapping
        context_length: Length of generated sequences
        steps: Number of denoising steps
        device: torch device
        eps: Small epsilon for numerical stability
        output_path: Path to save the GIF
        frame_duration: Duration of each frame in milliseconds
        image_width: Width of output images
        image_height: Height of output images
        save_every: Save a frame every N steps (1 = save all frames)

    Returns:
        Path to the generated GIF
    """
    model.eval()

    print(f"Generating animation with {steps} denoising steps...")
    print(f"Saving frames every {save_every} step(s)")

    step_size = (1 - eps) / steps
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    frames = []

    with torch.no_grad():
        # Start with random tokens
        x = torch.randint(0, vocab_size, (1, context_length), device=device)

        # Save initial random state
        text = decode(x[0], itos)
        step_info = f"Step 0/{steps} | σ = {noise(torch.ones(1, 1, device=device) * 1.0)[0].item():.4f} | Initial Random Noise"
        frame = text_to_image(text, image_width, image_height, step_info=step_info)
        frames.append(frame)
        print(f"  Frame 0/{steps} - Initial random noise")

        # Denoising loop
        for i in range(steps + 1):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            curr_sigma_bar = noise(t)[0]

            if i < steps:
                next_sigma_bar = noise(t - step_size)[0]
                delta_sigma = curr_sigma_bar - next_sigma_bar

                log_score = model(x, curr_sigma_bar)
                score = torch.exp(log_score)

                stag_score = staggered_score(score, delta_sigma)
                probs = stag_score * transition(x, delta_sigma, vocab_size)
                x = sample_categorical(probs)
            else:
                # Last denoising step
                delta_sigma = curr_sigma_bar

                log_score = model(x, curr_sigma_bar)
                score = torch.exp(log_score)

                stag_score = staggered_score(score, delta_sigma)
                probs = stag_score * transition(x, delta_sigma, vocab_size)
                x = sample_categorical(probs)

            # Save frame every N steps or at the end
            if (i + 1) % save_every == 0 or i == steps:
                text = decode(x[0], itos)
                sigma_val = curr_sigma_bar.item()

                if i == steps:
                    step_info = f"Step {i+1}/{steps} | σ = {sigma_val:.4f} | FINAL OUTPUT"
                else:
                    step_info = f"Step {i+1}/{steps} | σ = {sigma_val:.4f} | Denoising..."

                frame = text_to_image(text, image_width, image_height, step_info=step_info)
                frames.append(frame)
                print(f"  Frame {len(frames)-1}/{steps//save_every + 1} - Step {i+1}")

        # Add a few duplicate frames at the end so viewers can read the final text
        final_frame = frames[-1]
        for _ in range(5):
            frames.append(final_frame)

    # Save as animated GIF
    print(f"\nSaving animation to: {output_path}")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration,
        loop=0,
        optimize=False,
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Animation saved! ({len(frames)} frames, {file_size_mb:.2f} MB)")
    print(f"Duration: {len(frames) * frame_duration / 1000:.1f} seconds")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Generate animated GIF of discrete diffusion denoising process'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    parser.add_argument('--vocab', type=str, default=None,
                       help='Path to vocabulary pickle file')
    parser.add_argument('--steps', type=int, default=128,
                       help='Number of denoising steps (default: 128)')
    parser.add_argument('--output', type=str, default='outputs/denoising_animation.gif',
                       help='Output path for the GIF')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--width', type=int, default=800,
                       help='Image width (default: 800)')
    parser.add_argument('--height', type=int, default=600,
                       help='Image height (default: 600)')
    parser.add_argument('--duration', type=int, default=200,
                       help='Frame duration in milliseconds (default: 200)')
    parser.add_argument('--save-every', type=int, default=1,
                       help='Save a frame every N steps (default: 1 = all frames)')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Set random seed
    set_seed(args.seed)

    # Determine device
    if args.device:
        device = torch.device(args.device)
    elif config['device'] == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])

    print(f"Using device: {device}")

    # Load model checkpoint
    if not os.path.exists(args.model):
        print(f"Error: Model checkpoint not found: {args.model}")
        sys.exit(1)

    print(f"Loading model from: {args.model}")
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    # Extract model config
    if 'config' in checkpoint:
        model_config_dict = checkpoint['config']
        model_config = GPTConfig(**model_config_dict)
    else:
        print("Warning: Model config not found in checkpoint, using config.yaml")
        model_config = GPTConfig(
            block_size=config['model']['context_length'],
            vocab_size=checkpoint.get('vocab_size', 65),
            n_layer=config['model']['n_layer'],
            n_head=config['model']['n_head'],
            n_embd=config['model']['n_embd'],
            cond_dim=config['model']['cond_dim'],
            dropout=0.0,
            bias=config['model']['bias'],
        )

    vocab_size = checkpoint.get('vocab_size', model_config.vocab_size)

    # Initialize model
    model = GPT(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model parameters: {model.get_num_params() / 1e6:.2f}M")

    # Load vocabulary
    if args.vocab:
        vocab_path = args.vocab
    else:
        # Infer vocab path from model name
        model_name = os.path.splitext(os.path.basename(args.model))[0]
        if '_epoch_' in model_name:
            model_name = model_name.split('_epoch_')[0]
        vocab_path = os.path.join(config['paths']['vocab_dir'], f"{model_name}_vocab.pkl")

    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file not found: {vocab_path}")
        print("Please specify vocabulary file with --vocab")
        sys.exit(1)

    print(f"Loading vocabulary from: {vocab_path}")
    import pickle
    with open(vocab_path, 'rb') as f:
        vocab_meta = pickle.load(f)
        itos = vocab_meta['itos']

    # Initialize noise schedule
    noise = GeometricNoise(
        sigma_min=config['noise']['sigma_min'],
        sigma_max=config['noise']['sigma_max'],
    )

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Generate animation
    output_path = generate_animation(
        model=model,
        noise=noise,
        vocab_size=vocab_size,
        itos=itos,
        context_length=model_config.block_size,
        steps=args.steps,
        device=device,
        output_path=args.output,
        frame_duration=args.duration,
        image_width=args.width,
        image_height=args.height,
        save_every=args.save_every,
    )

    print(f"\n{'='*80}")
    print(f"Animation generation complete!")
    print(f"Output: {output_path}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
