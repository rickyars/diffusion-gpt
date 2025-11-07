"""
Helper script to create vocabulary file from a dataset.

This is useful if you have a trained model but the vocabulary file is missing.
The vocabulary file is normally created automatically during training.

USAGE:
    python create_vocab.py --dataset datasets/shakespeare.txt --output vocab/shakespeare_vocab.pkl
"""

import argparse
import os
import pickle
import sys
from collections import Counter


def create_vocabulary(data_path: str, output_path: str):
    """
    Create vocabulary from text file and save to pickle.

    Args:
        data_path: Path to .txt file
        output_path: Path to save vocabulary .pkl file
    """
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found: {data_path}")
        sys.exit(1)

    print(f"Reading dataset: {data_path}")
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"Dataset size: {len(text):,} characters")

    # Get unique characters and sort them
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Characters: {repr(''.join(chars))}")

    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Create vocabulary metadata
    vocab_meta = {
        'stoi': stoi,
        'itos': itos,
        'vocab_size': vocab_size
    }

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(vocab_meta, f)

    print(f"\n✓ Vocabulary saved to: {output_path}")
    print(f"  - vocab_size: {vocab_size}")
    print(f"  - stoi: dict mapping characters to indices")
    print(f"  - itos: dict mapping indices to characters")

    return vocab_meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vocabulary file from dataset')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to dataset .txt file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for vocabulary .pkl file')

    args = parser.parse_args()

    create_vocabulary(args.dataset, args.output)
