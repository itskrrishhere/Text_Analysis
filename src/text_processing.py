import os
from pathlib import Path
import pandas as pd
import nltk
import re
import emoji
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import text2emotion as te
from nltk.tokenize import sent_tokenize
from functools import lru_cache
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm  # For progress tracking

# Download required NLTK resources
nltk.download('punkt', quiet=True)

# -------------------------------
# Define an Optimized Emoji Regex Pattern
# -------------------------------
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
    "\U0001F1E0-\U0001F1FF"  # Flags (country codes)
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"  # Enclosed characters
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U0001F018-\U0001F270"  # Various symbols
    "\U0001F650-\U0001F67F"  # Ornamental dingbats
    "\U0001F700-\U0001F77F"  # Alchemical Symbols
    "\U0001F780-\U0001F7FF"  # Geometric Shapes
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows
    "\U0001F900-\U0001F9FF"  # Additional emoji
    "\U0001FA00-\U0001FA6F"  # Extended Symbols
    "\U0001FA70-\U0001FAFF"  # More emoji additions
    "\U0001FAB0-\U0001FABF"  # More animals
    "\U0001FAC0-\U0001FACF"  # More people
    "\U0001FAD0-\U0001FADF"  # More food
    "]+", flags=re.UNICODE
)

# -------------------------------
# Initialize Sentiment Analyzer
# -------------------------------
analyzer = SentimentIntensityAnalyzer()

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
def load_tweets(csv_file, chunksize):
    """
    Load tweets from a CSV file in chunks.
    Expects CSV to have at least 'id' and 'text' columns.
    """
    return pd.read_csv(csv_file, chunksize=chunksize)

def count_tweets(csv_file):
    """
    Count the number of tweets (rows) in the CSV file (excluding header).
    """
    with open(csv_file, "r", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1

# -------------------------------
# Sentence and Chunk Tokenization
# -------------------------------
def split_into_sentences(text):
    return sent_tokenize(text)

def split_sentence_into_chunks(sentence):
    tokens = sentence.split()
    n = len(tokens)
    if n < 3:
        return [sentence]
    chunk_size = n // 3
    return [
        " ".join(tokens[:chunk_size]),
        " ".join(tokens[chunk_size:2 * chunk_size]),
        " ".join(tokens[2 * chunk_size:])
    ]

# -------------------------------
# Emoji Detection and Placement
# -------------------------------
def detect_emojis_in_text(text):
    return [(match.group(), match.start()) for match in emoji_pattern.finditer(text)]

def determine_emoji_placement(chunks):
    placements = []
    for chunk in chunks:
        emojis = detect_emojis_in_text(chunk)
        if emojis:
            chunk_length = len(chunk)
            chunk_labels = []
            for _, pos in emojis:
                if pos < chunk_length / 3:
                    chunk_labels.append("start")
                elif pos < 2 * chunk_length / 3:
                    chunk_labels.append("middle")
                else:
                    chunk_labels.append("end")
            placements.append(chunk_labels)
        else:
            placements.append(["none"])
    return placements

# -------------------------------
# Text Preprocessing with Caching
# -------------------------------
@lru_cache(maxsize=10000)
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='')

# -------------------------------
# Sentiment Analysis
# -------------------------------
def get_sentiment_scores(text):
    return analyzer.polarity_scores(text)

def analyze_sentiment(text):
    score_with = get_sentiment_scores(text)
    score_without = get_sentiment_scores(remove_emojis(text))
    return score_with, score_without

# -------------------------------
# Emotion Detection
# -------------------------------
def get_dominant_emotion(text):
    emotions = te.get_emotion(text)
    if emotions:
        return max(emotions, key=emotions.get)
    return None

# -------------------------------
# Processing Individual Tweets
# -------------------------------
def process_tweet(tweet_id, tweet_text):
    sentences = split_into_sentences(tweet_text)
    tweet_result = {"tweet_id": tweet_id, "sentences": []}

    for sentence in sentences:
        sentence_dict = {"sentence": sentence, "chunks": []}
        sent_sentiment_with, sent_sentiment_without = analyze_sentiment(sentence)
        sent_emotion_with = get_dominant_emotion(sentence)
        sent_emotion_without = get_dominant_emotion(remove_emojis(sentence))
        sentence_dict["sentiment_with_emoji"] = sent_sentiment_with
        sentence_dict["sentiment_without_emoji"] = sent_sentiment_without
        sentence_dict["dominant_emotion_with_emoji"] = sent_emotion_with
        sentence_dict["dominant_emotion_without_emoji"] = sent_emotion_without

        chunks = split_sentence_into_chunks(sentence)
        placements = determine_emoji_placement(chunks)

        for i, chunk in enumerate(chunks):
            sentiment_with, sentiment_without = analyze_sentiment(chunk)
            emotion_with = get_dominant_emotion(chunk)
            emotion_without = get_dominant_emotion(remove_emojis(chunk))
            chunk_dict = {
                "chunk_text": chunk,
                "emoji_placements": placements[i],
                "sentiment_with_emoji": sentiment_with,
                "sentiment_without_emoji": sentiment_without,
                "dominant_emotion_with_emoji": emotion_with,
                "dominant_emotion_without_emoji": emotion_without
            }
            sentence_dict["chunks"].append(chunk_dict)
        tweet_result["sentences"].append(sentence_dict)
    return tweet_result

# -------------------------------
# Incremental Aggregation Functions
# -------------------------------
def update_aggregator(aggregator, tweet_result):
    aggregator["total_tweets_processed"] += 1
    for sentence in tweet_result["sentences"]:
        for chunk in sentence["chunks"]:
            for label in chunk["emoji_placements"]:
                aggregator["emoji_placement_distribution"][label] += 1
            comp_with = chunk["sentiment_with_emoji"].get("compound", 0)
            comp_without = chunk["sentiment_without_emoji"].get("compound", 0)
            aggregator["sentiment_diff_sum"] += (comp_with - comp_without)
            aggregator["total_chunks_processed"] += 1
            if chunk["dominant_emotion_with_emoji"]:
                aggregator["dominant_emotion_distribution_with_emoji"][chunk["dominant_emotion_with_emoji"]] += 1
            if chunk["dominant_emotion_without_emoji"]:
                aggregator["dominant_emotion_distribution_without_emoji"][chunk["dominant_emotion_without_emoji"]] += 1

# -------------------------------
# Main Execution Function with a Persistent Process Pool
# -------------------------------
def main():
    input_file = Path("cleaned_data/all_cleaned_tweets.csv")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "tweet_analysis_results.jsonl"
    insights_file = output_dir / "analysis_insights.json"

    total_tweets = count_tweets(input_file)

    aggregator = {
        "total_tweets_processed": 0,
        "total_chunks_processed": 0,
        "emoji_placement_distribution": Counter(),
        "sentiment_diff_sum": 0.0,
        "dominant_emotion_distribution_with_emoji": Counter(),
        "dominant_emotion_distribution_without_emoji": Counter()
    }

    # Clear previous results.
    with open(results_file, "w", encoding="utf-8") as outfile:
        pass

    chunksize = 7500
    reader = load_tweets(input_file, chunksize)
    global_pbar = tqdm(total=total_tweets, desc="Processing tweets")

    # Create a persistent process pool that is used throughout the entire file processing.
    max_workers = max(1, os.cpu_count() * 2) if hasattr(os, 'cpu_count') else 30
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Process CSV file in chunks.
        for chunk in reader:
            if 'text' not in chunk.columns:
                raise ValueError("CSV file must contain at least a 'text' column")
            if 'id' not in chunk.columns:
                chunk['id'] = range(1, len(chunk) + 1)
            tweets_to_process = list(zip(chunk['id'], chunk['text']))

            # Submit all tweets for processing concurrently using the persistent pool.
            futures = {executor.submit(process_tweet, tweet_id, tweet_text): (tweet_id, tweet_text)
                       for tweet_id, tweet_text in tweets_to_process}
            chunk_results = []
            for future in as_completed(futures):
                tweet_id, tweet_text = futures[future]
                try:
                    tweet_result = future.result()
                    chunk_results.append(tweet_result)
                    update_aggregator(aggregator, tweet_result)
                except Exception as exc:
                    print(f"Tweet {tweet_id} processing generated an exception: {exc}")
                finally:
                    global_pbar.update(1)

            # Write the processed results for the current chunk.
            with open(results_file, "a", encoding="utf-8") as outfile:
                for tweet_result in chunk_results:
                    outfile.write(json.dumps(tweet_result) + "\n")
            # Free memory from the current chunk.
            del chunk_results

    global_pbar.close()

    total_chunks = aggregator["total_chunks_processed"]
    avg_sentiment_diff = (aggregator["sentiment_diff_sum"] / total_chunks) if total_chunks > 0 else 0
    aggregated_insights = {
        "total_tweets_processed": aggregator["total_tweets_processed"],
        "total_chunks_processed": total_chunks,
        "emoji_placement_distribution": dict(aggregator["emoji_placement_distribution"]),
        "average_sentiment_compound_difference": avg_sentiment_diff,
        "dominant_emotion_distribution_with_emoji": dict(aggregator["dominant_emotion_distribution_with_emoji"]),
        "dominant_emotion_distribution_without_emoji": dict(aggregator["dominant_emotion_distribution_without_emoji"])
    }

    with open(insights_file, "w", encoding="utf-8") as outfile:
        json.dump(aggregated_insights, outfile, indent=2, ensure_ascii=False)

    print(f"Analysis complete. Processed {aggregator['total_tweets_processed']} tweets.")
    print(f"Detailed results saved to {results_file}")
    print(f"Aggregated insights saved to {insights_file}")

if __name__ == '__main__':
    main()
