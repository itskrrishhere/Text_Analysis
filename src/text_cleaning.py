import os
import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Precompiled regex patterns
RE_HTML_AMP = re.compile(r'&amp;?', re.IGNORECASE)
RE_MENTION = re.compile(r'@\w+')
RE_HASHTAG = re.compile(r'#\w+')
RE_URL = re.compile(r'http\S+')
RE_SPACES = re.compile(r'\s+')

def clean_tweet(text: str) -> str:
    """Clean tweet text by removing mentions, hashtags, URLs, etc."""
    if not isinstance(text, str):
        return ""
    text = RE_HTML_AMP.sub('&', text)
    text = RE_MENTION.sub('', text)
    text = RE_HASHTAG.sub('', text)
    text = RE_URL.sub('', text)
    text = RE_SPACES.sub(' ', text).strip()
    return text

def read_csv_with_fallback(file_path: str, encodings=None) -> pd.DataFrame:
    """Try reading CSV file with multiple encodings and flexible header detection."""
    if encodings is None:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'utf-16']

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip', engine='python')

            # Standardize column name if needed
            df.columns = [col.strip().lower() for col in df.columns]
            if 'text' not in df.columns:
                if len(df.columns) == 1:
                    df.columns = ['text']
                else:
                    continue  # Skip if no usable 'text' column

            return df[['text']]

        except Exception:
            continue

    # Fallback to line-by-line read
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = [line.strip() for line in f if line.strip()]
    return pd.DataFrame({'text': lines})


def process_file(file_path: Path) -> pd.DataFrame:
    """Process and clean a single CSV file."""
    try:
        df = read_csv_with_fallback(file_path)

        # Fix column name if it’s 'Text' or any variation
        df.columns = [col.lower() for col in df.columns]
        if 'text' not in df.columns:
            df.columns = ['text']  # fallback if it's a single column

        df['cleaned_text'] = df['text'].apply(clean_tweet)
        df = df[df['cleaned_text'].astype(bool)]  # Remove empty strings
        return df[['cleaned_text']].drop_duplicates()

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return pd.DataFrame(columns=['cleaned_text'])


def main():
    input_folder = Path("archive")
    output_folder = Path("cleaned_data")
    output_file = output_folder / "all_cleaned_tweets.csv"

    output_folder.mkdir(exist_ok=True)
    all_csv_files = list(input_folder.glob("*.csv"))

    print(f"Processing {len(all_csv_files)} files with multithreading...")

    dataframes = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_file, file_path): file_path for file_path in all_csv_files}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Cleaning tweets"):
            df = future.result()
            if not df.empty:
                dataframes.append(df)

    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.drop_duplicates(subset=['cleaned_text'], inplace=True)
        combined_df.rename(columns={'cleaned_text': 'text'}, inplace=True)
        combined_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"\n✅ Saved {len(combined_df)} cleaned tweets to {output_file}")
    else:
        print("\n⚠️ No valid data processed.")

if __name__ == "__main__":
    main()
