# CSV to Dataset Converter Guide

Convert CSV files to the `.txt` format required for training discrete diffusion models.

## Quick Start

```bash
# Using column name
python csv_to_dataset.py data.csv datasets/my_data.txt --column "text"

# Using column index (0-based)
python csv_to_dataset.py data.csv datasets/my_data.txt --column 0
```

## Usage

```bash
python csv_to_dataset.py INPUT_CSV OUTPUT_TXT --column COLUMN_NAME [OPTIONS]
```

### Required Arguments

- **`INPUT_CSV`**: Path to your CSV file
- **`OUTPUT_TXT`**: Where to save the output (e.g., `datasets/my_data.txt`)
- **`--column COLUMN`** or **`-c COLUMN`**: Column containing text
  - Can be column name (e.g., `"review"`, `"text"`, `"content"`)
  - Can be column index (e.g., `0`, `1`, `2` - zero-based)

### Optional Arguments

- **`--max-rows N`**: Only process first N rows (useful for large datasets)
- **`--no-skip-header`**: Don't skip first row (use if CSV has no header)
- **`--encoding ENCODING`**: File encoding (default: `utf-8`)

## Examples

### Example 1: Amazon Reviews

```bash
# CSV has columns: id, rating, review, date
python csv_to_dataset.py amazon_reviews.csv datasets/amazon_reviews.txt --column "review"
```

Output:
```
✓ Dataset created successfully!
  Output: datasets/amazon_reviews.txt
  Rows written: 50,000
  Rows skipped (empty): 123
  File size: 12.3 MB
```

### Example 2: Using Column Index

```bash
# If you don't know column names, use index (0 = first column)
python csv_to_dataset.py data.csv datasets/my_data.txt --column 2
```

This uses the 3rd column (index 2).

### Example 3: Limit Rows for Testing

```bash
# Only process first 1,000 rows for quick testing
python csv_to_dataset.py big_dataset.csv datasets/test.txt --column "text" --max-rows 1000
```

### Example 4: CSV Without Header

```bash
# If your CSV doesn't have a header row
python csv_to_dataset.py no_header.csv datasets/output.txt --column 0 --no-skip-header
```

### Example 5: Different Encoding

```bash
# For files with non-UTF8 encoding
python csv_to_dataset.py latin1_data.csv datasets/output.txt --column "text" --encoding latin1
```

## Common Use Cases

### GitHub Commits

```bash
# CSV columns: repo, commit_hash, message, author
python csv_to_dataset.py commits.csv datasets/github_commits.txt --column "message"
```

### Reddit Comments

```bash
# CSV columns: subreddit, author, body, score
python csv_to_dataset.py reddit.csv datasets/reddit_comments.txt --column "body"
```

### Product Reviews

```bash
# CSV columns: product_id, title, review_text, stars
python csv_to_dataset.py reviews.csv datasets/reviews.txt --column "review_text"
```

### News Articles

```bash
# CSV columns: date, headline, article, category
python csv_to_dataset.py news.csv datasets/news_articles.txt --column "article"
```

### Tweets/Social Media

```bash
# CSV columns: user_id, tweet_text, likes, retweets
python csv_to_dataset.py tweets.csv datasets/twitter_posts.txt --column "tweet_text"
```

## Output Format

The script creates a `.txt` file with:
- **One line per CSV row**
- **Only the text column** (other columns ignored)
- **Empty rows skipped**
- **UTF-8 encoding**

Example:
```
This is the first review text
This is the second review text
This is the third review text
```

## Tips

### Finding Column Names

If you don't know your column names:

```bash
# Run without specifying output to see available columns
python csv_to_dataset.py data.csv test.txt --column "invalid_name"
```

Error message will show:
```
Available columns: ['id', 'text', 'label', 'date']
```

### Handling Large Files

For large CSV files (>1GB):

```bash
# Process in chunks
python csv_to_dataset.py big.csv datasets/part1.txt --column "text" --max-rows 100000

# Then combine later if needed
cat datasets/part*.txt > datasets/combined.txt
```

### Multiple Text Columns

If you want to use multiple columns:

```bash
# Extract each column separately
python csv_to_dataset.py data.csv datasets/titles.txt --column "title"
python csv_to_dataset.py data.csv datasets/bodies.txt --column "body"

# Combine them
cat datasets/titles.txt datasets/bodies.txt > datasets/all_text.txt
```

## After Conversion

Once you have your `.txt` file:

1. **Add to config.yaml**:
```yaml
datasets:
  my_dataset:
    path: datasets/my_data.txt
    enabled: true
    description: "My custom dataset"
```

2. **Train a model**:
```bash
python train.py --dataset my_dataset
```

## Troubleshooting

### "Column 'text' not found"

Check your column names:
```bash
head -1 data.csv
```

Use the exact column name (case-sensitive) or use column index instead.

### "File too large"

Use `--max-rows` to process in chunks:
```bash
python csv_to_dataset.py huge.csv datasets/sample.txt --column "text" --max-rows 10000
```

### "Encoding errors"

Try different encoding:
```bash
python csv_to_dataset.py data.csv output.txt --column "text" --encoding iso-8859-1
```

Common encodings: `utf-8`, `latin1`, `iso-8859-1`, `cp1252`, `utf-16`

### "Empty output file"

- Check if your text column actually has data
- Make sure you're using the right column name/index
- Try with `--no-skip-header` if your CSV doesn't have a header

## Built-in Help

For full options:
```bash
python csv_to_dataset.py --help
```

## Data Quality Tips

For best training results:
- **Minimum**: ~10,000 lines of text
- **Recommended**: 100,000+ lines
- **Text length**: 50-500 characters per line works well
- **Remove duplicates**: Deduplicate your CSV first if needed
- **Clean data**: Remove HTML tags, special characters, etc. if necessary
