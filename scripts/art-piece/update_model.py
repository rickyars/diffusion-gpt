#!/usr/bin/env python3
"""
Update model data in we.html from we.template.html

This script fills the template with new model data and generates we.html.
It reads from the template, loads the model and vocabulary, and injects them
into the HTML while preserving all your manual edits.

Usage:
    python update_model.py --model path/to/model.onnx --dataset dataset_name

    python update_model.py
        (uses defaults: models/confessions_model.onnx, confessions)
"""

import argparse
import json
import base64
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, 'we.template.html')
OUTPUT_PATH = os.path.join(SCRIPT_DIR, 'we.html')


def load_vocab(dataset_name):
    """Load vocabulary JSON file."""
    vocab_path = os.path.join(PROJECT_ROOT, 'vocab', f'{dataset_name}_vocab.json')
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")

    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    if 'itos' not in vocab or 'stoi' not in vocab:
        raise ValueError(f"Vocabulary missing 'itos' and 'stoi' keys: {vocab_path}")

    return vocab


def load_model_base64(model_path):
    """Load ONNX model and encode to base64."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    with open(model_path, 'rb') as f:
        onnx_bytes = f.read()

    return base64.b64encode(onnx_bytes).decode('utf-8')


def extract_model_config(onnx_bytes):
    """
    Extract model config from ONNX model metadata.
    Falls back to sensible defaults if metadata is not available.
    """
    # For now, return hardcoded config
    # In future, could parse ONNX model graph to extract dimensions
    return {
        "block_size": 256,
        "vocab_size": 100,
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "cond_dim": 64
    }


def load_inference_engine():
    """Load the inference engine JavaScript code."""
    inference_path = os.path.join(SCRIPT_DIR, 'onnx_inference_engine.js')
    if not os.path.exists(inference_path):
        raise FileNotFoundError(f"Inference engine not found: {inference_path}")

    with open(inference_path, 'r', encoding='utf-8') as f:
        return f.read()


def check_for_unapplied_edits(template_content):
    """
    Check if we.html has been manually edited since template was last extracted.
    Returns True if unapplied edits are detected.

    Strategy: Since the template contains placeholders like {{MODEL_CONFIG}},
    we fill them with a known pattern and compare against the current we.html
    to see if the structure matches. If it matches, the current we.html was
    generated from the current template, so no unapplied edits.
    """
    import re

    if not os.path.exists(OUTPUT_PATH):
        # File doesn't exist yet, safe to create
        return False

    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        current_html = f.read()

    # Normalize both template and current HTML by replacing large data sections
    # This removes the model data, vocab, and inference engine code which change
    # but keep the structure the same.

    def normalize_structure(html_content):
        """Remove large data sections to compare just structure."""
        normalized = html_content
        # Replace all placeholder values (whether filled or not)
        normalized = re.sub(r'{{[^}]+}}', '{{PLACEHOLDER}}', normalized)
        # Replace filled model config (can be multiline)
        normalized = re.sub(r'const MODEL_CONFIG = \{[^}]*\};', 'const MODEL_CONFIG = {{PLACEHOLDER}};', normalized)
        # Replace filled vocab (very long JSON)
        normalized = re.sub(r'const VOCAB = \{[^}]*\};', 'const VOCAB = {{PLACEHOLDER}};', normalized, flags=re.DOTALL)
        # Replace base64 model data (huge)
        normalized = re.sub(r"const MODEL_BASE64 = '[^']*';", "const MODEL_BASE64 = '{{PLACEHOLDER}}';", normalized)
        # Replace inference engine code
        normalized = re.sub(r'<script id="inference-engine">[^<]*</script>', '<script id="inference-engine">{{PLACEHOLDER}}</script>', normalized, flags=re.DOTALL)
        # Replace any other script tags with code
        normalized = re.sub(r'<script[^>]*>.*?</script>', '<script>{{PLACEHOLDER}}</script>', normalized, flags=re.DOTALL)
        return normalized.strip()

    template_struct = normalize_structure(template_content)
    current_struct = normalize_structure(current_html)

    # If the normalized structures match, the current HTML matches the template
    # structure (just with filled-in data), so no unapplied edits exist
    if template_struct == current_struct:
        return False

    # If structures differ, there are custom edits in the current HTML that
    # are not in the template
    return True


def create_backup(file_path):
    """Create a backup of the file before overwriting."""
    if os.path.exists(file_path):
        backup_path = f"{file_path}.backup"
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return backup_path
    return None


def update_model(dataset_name, model_path, force=False):
    """Generate we.html from template with new model data."""

    print("Updating WE art piece with new model...")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_path}")
    print("="*60)

    # Load template
    if not os.path.exists(TEMPLATE_PATH):
        print(f"Error: Template not found: {TEMPLATE_PATH}")
        print("Run 'python extract_template.py' first to create the template.")
        return False

    print("\nLoading template...")
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        template = f.read()

    # Check for unapplied edits
    if check_for_unapplied_edits(template):
        if not force:
            print("\n" + "!"*60)
            print("WARNING: Unapplied edits detected in we.html!")
            print("!"*60)
            print("\nYou have made manual changes to we.html that are not")
            print("yet saved to the template. These changes would be lost.")
            print("\nTo save your edits:")
            print("  1. Run: python extract_template.py")
            print("  2. Review the changes in we.template.html")
            print("  3. Then run this script again: python update_model.py")
            print("\nOr to discard edits and continue:")
            print("  Run with --force flag: python update_model.py --force")
            print("!"*60)
            print("\nCreating backup: we.html.backup")
            backup_path = create_backup(OUTPUT_PATH)
            if backup_path:
                print(f"Your current we.html has been saved to: {backup_path}")
            return False
        else:
            print("\n--force flag detected. Proceeding despite unapplied edits.")
            backup_path = create_backup(OUTPUT_PATH)
            if backup_path:
                print(f"Created backup: {backup_path}")

    # Load vocabulary
    print("Loading vocabulary...")
    vocab = load_vocab(dataset_name)
    vocab_json = json.dumps(vocab)
    print(f"  Vocabulary size: {len(vocab.get('itos', {}))} characters")

    # Load model
    print("Loading ONNX model...")
    model_b64 = load_model_base64(model_path)
    model_size_mb = len(model_b64) / (1024 * 1024 * 1.33)
    print(f"  Model size: {model_size_mb:.2f} MB (base64 encoded)")

    # Extract model config
    print("Extracting model configuration...")
    model_config = extract_model_config(None)
    print(f"  Config: {model_config}")

    # Load inference engine
    print("Loading inference engine...")
    inference_engine = load_inference_engine()
    print(f"  Engine size: {len(inference_engine)} bytes")

    # Fill template
    print("\nGenerating HTML...")
    html = template
    html = html.replace('{{MODEL_CONFIG}}', json.dumps(model_config))
    html = html.replace('{{VOCAB}}', vocab_json)
    html = html.replace('{{MODEL_BASE64}}', model_b64)
    html = html.replace('{{INFERENCE_ENGINE}}', inference_engine)

    # Write output
    print(f"Writing {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        f.write(html)

    output_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)

    print("\n" + "="*60)
    print("MODEL UPDATE COMPLETE!")
    print("="*60)
    print(f"File: {OUTPUT_PATH}")
    print(f"Size: {output_size:.2f} MB")
    print("\nNext steps:")
    print("  1. Open we.html in a browser to test")
    print("  2. If needed, edit we.html and run 'python extract_template.py'")
    print("  3. The updated we.template.html is ready to commit")
    print("="*60)

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Update WE art piece with new model data'
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
        help='Path to ONNX model (default: models/{dataset}_model.onnx)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force update even if unapplied edits are detected (creates backup)'
    )

    args = parser.parse_args()

    # Generate default model path if not specified
    if args.model is None:
        args.model = os.path.join(PROJECT_ROOT, 'models', f'{args.dataset}_model.onnx')

    success = update_model(args.dataset, args.model, force=args.force)
    sys.exit(0 if success else 1)
