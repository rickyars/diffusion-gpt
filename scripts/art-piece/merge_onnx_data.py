"""
Merge ONNX model with external data into a single file.
"""

import argparse
import os
import onnx
from onnx.external_data_helper import convert_model_to_external_data, convert_model_from_external_data

def merge_onnx(input_path, output_path):
    """Merge ONNX model with external .data file into single file."""
    print(f"Loading ONNX model from {input_path}...")

    # Load the model (it will automatically load the .data file)
    model = onnx.load(input_path)

    print("Converting external data to embedded...")

    # Convert external data to embedded
    convert_model_from_external_data(model)

    print(f"Saving merged model to {output_path}...")
    onnx.save(model, output_path)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Merged model size: {size_mb:.2f} MB")
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Merge ONNX model with external .data file into single file'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to ONNX model with external data (e.g., models/confessions_model.onnx)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for merged model (default: adds _merged suffix)'
    )

    args = parser.parse_args()

    # Generate output path if not specified
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_merged{ext}"

    merge_onnx(args.input, args.output)
