"""
Extract abstracts from arXiv dataset for training.
The arXiv dataset from Kaggle is in JSON format (one JSON object per line).
This script extracts the 'abstract' field and saves to a .txt file.
"""

import argparse
import json
import os
import sys
from tqdm import tqdm


def extract_abstracts(
    json_path: str,
    output_path: str,
    max_abstracts: int = None,
    min_length: int = 50,
    max_length: int = None,
    categories: list = None,
):
    """
    Extract abstracts from arXiv JSON dataset.

    Args:
        json_path: Path to input JSON file (arXiv metadata)
        output_path: Path to output .txt file
        max_abstracts: Maximum number of abstracts to extract (default: None = all)
        min_length: Minimum abstract length in characters (default: 50)
        max_length: Maximum abstract length in characters (default: None = no limit)
        categories: List of arXiv categories to filter (e.g., ['cs.AI', 'cs.LG'])
    """
    print(f"Reading arXiv dataset from: {json_path}")

    if not os.path.exists(json_path):
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    # Count total lines for progress bar
    print("Counting entries...")
    with open(json_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    print(f"Total entries in dataset: {total_lines:,}")

    abstracts_written = 0
    abstracts_skipped = 0
    parse_errors = 0

    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    try:
        with open(json_path, 'r', encoding='utf-8', errors='ignore') as in_file, \
             open(output_path, 'w', encoding='utf-8') as out_file:

            for line in tqdm(in_file, total=total_lines, desc="Extracting abstracts"):
                if max_abstracts and abstracts_written >= max_abstracts:
                    break

                try:
                    # Parse JSON line
                    paper = json.loads(line.strip())

                    # Extract abstract
                    abstract = paper.get('abstract', '').strip()

                    # Skip if no abstract
                    if not abstract:
                        abstracts_skipped += 1
                        continue

                    # Filter by category if specified
                    if categories:
                        paper_categories = paper.get('categories', '')
                        # Categories are typically space-separated
                        if not any(cat in paper_categories for cat in categories):
                            abstracts_skipped += 1
                            continue

                    # Filter by length
                    if len(abstract) < min_length:
                        abstracts_skipped += 1
                        continue

                    if max_length and len(abstract) > max_length:
                        abstract = abstract[:max_length]

                    # Clean abstract: remove newlines and extra whitespace
                    abstract = ' '.join(abstract.split())

                    # Write to output file (one abstract per line)
                    out_file.write(abstract + '\n')
                    abstracts_written += 1

                except json.JSONDecodeError:
                    parse_errors += 1
                    continue
                except Exception as e:
                    parse_errors += 1
                    continue

    except Exception as e:
        print(f"\nError processing file: {e}")
        sys.exit(1)

    print(f"\n✓ Abstracts extracted successfully!")
    print(f"  Output: {output_path}")
    print(f"  Abstracts written: {abstracts_written:,}")
    print(f"  Abstracts skipped: {abstracts_skipped:,}")
    if parse_errors > 0:
        print(f"  Parse errors: {parse_errors:,}")

    # Show file size and stats
    file_size = os.path.getsize(output_path)
    if file_size < 1024:
        size_str = f"{file_size} bytes"
    elif file_size < 1024 * 1024:
        size_str = f"{file_size / 1024:.1f} KB"
    elif file_size < 1024 * 1024 * 1024:
        size_str = f"{file_size / (1024 * 1024):.1f} MB"
    else:
        size_str = f"{file_size / (1024 * 1024 * 1024):.2f} GB"

    print(f"  File size: {size_str}")

    if abstracts_written > 0:
        # Calculate average abstract length
        with open(output_path, 'r', encoding='utf-8') as f:
            total_chars = sum(len(line) for line in f)
        avg_length = total_chars / abstracts_written
        print(f"  Average abstract length: {avg_length:.0f} characters")


def main():
    parser = argparse.ArgumentParser(
        description='Extract abstracts from arXiv dataset for training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all abstracts
  python extract_arxiv_abstracts.py arxiv-metadata-oai-snapshot.json datasets/arxiv_abstracts.txt

  # Extract first 100,000 abstracts
  python extract_arxiv_abstracts.py arxiv-metadata-oai-snapshot.json datasets/arxiv_abstracts.txt --max 100000

  # Extract only CS (Computer Science) papers
  python extract_arxiv_abstracts.py arxiv-metadata-oai-snapshot.json datasets/arxiv_cs_abstracts.txt --categories cs.AI cs.LG cs.CL

  # Filter by length (min 100, max 1000 characters)
  python extract_arxiv_abstracts.py arxiv-metadata-oai-snapshot.json datasets/arxiv_abstracts.txt --min-length 100 --max-length 1000

Notes:
  - The arXiv dataset is typically named 'arxiv-metadata-oai-snapshot.json'
  - Download it from: https://www.kaggle.com/datasets/Cornell-University/arxiv
  - The file is large (~3.5 GB), so processing may take a few minutes
  - Use --max to limit the number of abstracts for testing
        """
    )

    parser.add_argument('json_path',
                       help='Path to arXiv JSON file (arxiv-metadata-oai-snapshot.json)')
    parser.add_argument('output_path',
                       help='Path to output .txt file (e.g., datasets/arxiv_abstracts.txt)')
    parser.add_argument('--max', type=int, default=None,
                       help='Maximum number of abstracts to extract (default: all)')
    parser.add_argument('--min-length', type=int, default=50,
                       help='Minimum abstract length in characters (default: 50)')
    parser.add_argument('--max-length', type=int, default=None,
                       help='Maximum abstract length in characters (default: no limit)')
    parser.add_argument('--categories', nargs='+', default=None,
                       help='Filter by arXiv categories (e.g., cs.AI cs.LG cs.CL)')

    args = parser.parse_args()

    extract_abstracts(
        json_path=args.json_path,
        output_path=args.output_path,
        max_abstracts=args.max,
        min_length=args.min_length,
        max_length=args.max_length,
        categories=args.categories,
    )


if __name__ == '__main__':
    main()
