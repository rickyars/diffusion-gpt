#!/usr/bin/env python3
"""
Extract we.template.html from we.html

This script reads the current we.html file and extracts it into a template
with clearly marked data injection points. The template preserves all your
manual edits while making it easy to update model data independently.

Usage:
    python extract_template.py

This creates/updates we.template.html with placeholders for:
    - {{MODEL_CONFIG}}
    - {{VOCAB}}
    - {{MODEL_BASE64}}
    - {{INFERENCE_ENGINE}}
"""

import re
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WE_HTML_PATH = os.path.join(SCRIPT_DIR, 'we.html')
TEMPLATE_PATH = os.path.join(SCRIPT_DIR, 'we.template.html')


def extract_template():
    """Read we.html and create we.template.html with data placeholders."""

    if not os.path.exists(WE_HTML_PATH):
        print(f"Error: {WE_HTML_PATH} not found")
        return False

    print(f"Reading {WE_HTML_PATH}...")
    with open(WE_HTML_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    print("Extracting data sections...")

    # Extract MODEL_CONFIG
    model_config_match = re.search(
        r'const MODEL_CONFIG = (\{[^}]*\});',
        content
    )
    if not model_config_match:
        print("Error: Could not find MODEL_CONFIG")
        return False

    # Extract VOCAB
    vocab_match = re.search(
        r'const VOCAB = (\{.*?\});',
        content,
        re.DOTALL
    )
    if not vocab_match:
        print("Error: Could not find VOCAB")
        return False

    # Extract MODEL_BASE64 - this is tricky because it's a very long string
    model_b64_match = re.search(
        r"const MODEL_BASE64 = '([^']{100,})';",
        content
    )
    if not model_b64_match:
        print("Error: Could not find MODEL_BASE64")
        return False

    # Extract inference engine code
    # Find from "// ============" before "// INFERENCE ENGINE" to "// ============" before "// MAIN APPLICATION"
    inference_match = re.search(
        r'// ============================================================================\s*// INFERENCE ENGINE\s*// ============================================================================\s*(.*?)\s*// ============================================================================\s*// MAIN APPLICATION',
        content,
        re.DOTALL
    )
    if not inference_match:
        print("Error: Could not find INFERENCE ENGINE section")
        return False

    inference_code = inference_match.group(1).strip()

    # Replace data sections with placeholders
    template_content = content

    # Replace MODEL_CONFIG
    template_content = re.sub(
        r'const MODEL_CONFIG = \{[^}]*\};',
        'const MODEL_CONFIG = {{MODEL_CONFIG}};',
        template_content
    )

    # Replace VOCAB
    template_content = re.sub(
        r'const VOCAB = \{.*?\};',
        'const VOCAB = {{VOCAB}};',
        template_content,
        flags=re.DOTALL
    )

    # Replace MODEL_BASE64
    template_content = re.sub(
        r"const MODEL_BASE64 = '[^']*';",
        "const MODEL_BASE64 = '{{MODEL_BASE64}}';",
        template_content
    )

    # Replace inference engine code
    template_content = re.sub(
        r'// ============================================================================\s*// INFERENCE ENGINE\s*// ============================================================================\s*.*?\s*// ============================================================================\s*// MAIN APPLICATION',
        '''// ============================================================================
// INFERENCE ENGINE
// ============================================================================

{{INFERENCE_ENGINE}}

// ============================================================================
// MAIN APPLICATION''',
        template_content,
        flags=re.DOTALL
    )

    # Write template
    print(f"Writing {TEMPLATE_PATH}...")
    with open(TEMPLATE_PATH, 'w', encoding='utf-8') as f:
        f.write(template_content)

    # Print summary
    print("\n" + "="*60)
    print("TEMPLATE EXTRACTED!")
    print("="*60)
    print(f"Template: {TEMPLATE_PATH}")
    print("\nData placeholders:")
    print("  {{MODEL_CONFIG}}    - Model architecture config")
    print("  {{VOCAB}}           - Character vocabulary (int-to-string mapping)")
    print("  {{MODEL_BASE64}}    - ONNX model (base64 encoded)")
    print("  {{INFERENCE_ENGINE}} - DiffusionInferenceEngine class")
    print("\nWorkflow:")
    print("  1. Edit we.html freely (styling, features, etc.)")
    print("  2. Run: python extract_template.py")
    print("  3. Commit we.template.html to git")
    print("  4. Run: python update_model.py --model <path> --dataset <name>")
    print("  5. Test the generated we.html in browser")
    print("="*60)

    return True


if __name__ == '__main__':
    success = extract_template()
    sys.exit(0 if success else 1)
