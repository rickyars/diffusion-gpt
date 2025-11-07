"""
Convert CSV file to training dataset format.
Extracts text from specified column and saves as .txt file (one line per row).
"""

import argparse
import csv
import os
import sys


def csv_to_dataset(
    csv_path: str,
    output_path: str,
    text_column: str,
    skip_header: bool = True,
    encoding: str = 'utf-8',
    max_rows: int = None,
):
    """
    Convert CSV file to dataset format.

    Args:
        csv_path: Path to input CSV file
        output_path: Path to output .txt file
        text_column: Name or index of column containing text
        skip_header: Whether to skip first row (default: True)
        encoding: File encoding (default: utf-8)
        max_rows: Maximum number of rows to process (default: None = all)
    """
    print(f"Reading CSV from: {csv_path}")
    print(f"Text column: {text_column}")

    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    rows_written = 0
    rows_skipped = 0

    try:
        with open(csv_path, 'r', encoding=encoding, errors='ignore') as csv_file:
            # Try to detect if text_column is an index or column name
            reader = csv.DictReader(csv_file)

            # Check if text_column is a valid column name
            if text_column not in reader.fieldnames:
                # Try as index
                try:
                    col_index = int(text_column)
                    print(f"Using column index: {col_index}")
                    # Reopen file to read by index
                    csv_file.seek(0)
                    reader = csv.reader(csv_file)
                    if skip_header:
                        next(reader)  # Skip header

                    use_index = True
                except ValueError:
                    print(f"Error: Column '{text_column}' not found in CSV")
                    print(f"Available columns: {reader.fieldnames}")
                    sys.exit(1)
            else:
                use_index = False
                print(f"Using column name: {text_column}")
                print(f"Available columns: {reader.fieldnames}")

            # Write to output file
            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as out_file:
                if use_index:
                    for row in reader:
                        if max_rows and rows_written >= max_rows:
                            break

                        if len(row) > col_index:
                            text = row[col_index].strip()
                            if text:  # Skip empty rows
                                out_file.write(text + '\n')
                                rows_written += 1
                            else:
                                rows_skipped += 1
                        else:
                            rows_skipped += 1
                else:
                    for row in reader:
                        if max_rows and rows_written >= max_rows:
                            break

                        text = row.get(text_column, '').strip()
                        if text:  # Skip empty rows
                            out_file.write(text + '\n')
                            rows_written += 1
                        else:
                            rows_skipped += 1

    except Exception as e:
        print(f"Error processing CSV: {e}")
        sys.exit(1)

    print(f"\n✓ Dataset created successfully!")
    print(f"  Output: {output_path}")
    print(f"  Rows written: {rows_written:,}")
    print(f"  Rows skipped (empty): {rows_skipped:,}")

    # Show file size
    file_size = os.path.getsize(output_path)
    if file_size < 1024:
        size_str = f"{file_size} bytes"
    elif file_size < 1024 * 1024:
        size_str = f"{file_size / 1024:.1f} KB"
    else:
        size_str = f"{file_size / (1024 * 1024):.1f} MB"
    print(f"  File size: {size_str}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert CSV file to training dataset format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using column name
  python csv_to_dataset.py data.csv datasets/my_data.txt --column "text"

  # Using column index (0-based)
  python csv_to_dataset.py data.csv datasets/my_data.txt --column 0

  # Limit to first 10,000 rows
  python csv_to_dataset.py data.csv datasets/my_data.txt --column "review" --max-rows 10000

  # Don't skip header (if CSV has no header row)
  python csv_to_dataset.py data.csv datasets/my_data.txt --column 0 --no-skip-header
        """
    )

    parser.add_argument('csv_path', help='Path to input CSV file')
    parser.add_argument('output_path', help='Path to output .txt file (e.g., datasets/my_data.txt)')
    parser.add_argument('--column', '-c', required=True,
                       help='Column name or index (0-based) containing text')
    parser.add_argument('--no-skip-header', action='store_true',
                       help='Do not skip first row (use if CSV has no header)')
    parser.add_argument('--encoding', default='utf-8',
                       help='File encoding (default: utf-8)')
    parser.add_argument('--max-rows', type=int, default=None,
                       help='Maximum number of rows to process (default: all)')

    args = parser.parse_args()

    csv_to_dataset(
        csv_path=args.csv_path,
        output_path=args.output_path,
        text_column=args.column,
        skip_header=not args.no_skip_header,
        encoding=args.encoding,
        max_rows=args.max_rows,
    )


if __name__ == '__main__':
    main()
