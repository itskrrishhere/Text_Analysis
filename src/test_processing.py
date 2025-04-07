import os
from pathlib import Path
import pandas as pd
import nltk
import re
import emoji
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import text2emotion as te
from nltk.tokenize import sent_tokenize, word_tokenize
from functools import lru_cache
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

# Download required NLTK resources
nltk.download('punkt', quiet=True)

# -------------------------------
# Define an Optimized Emoji Regex Pattern
# -------------------------------
emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons 🙂
    "\U0001F300-\U0001F5FF"  # Miscellaneous Symbols and Pictographs 🌈
    "\U0001F680-\U0001F6FF"  # Transport & Map Symbols 🚀
    "\U0001F1E0-\U0001F1FF"  # Flags (country codes) 🇮🇪
    "\U00002702-\U000027B0"  # Dingbats ✂️
    "\U000024C2-\U0001F251"  # Enclosed characters Ⓜ️
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs 🦄
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A 🛞
    "\U0001F018-\U0001F270"  # Various symbols 🔞
    "\U0001F650-\U0001F67F"  # Ornamental dingbats 🕊️
    "\U0001F700-\U0001F77F"  # Alchemical Symbols 🧪
    "\U0001F780-\U0001F7FF"  # Geometric Shapes 🟧
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows 🔀
    "\U0001F900-\U0001F9FF"  # Additional emoji (🦠, 🧑‍🤝‍🧑)
    "\U0001FA00-\U0001FA6F"  # Extended Symbols 🛗
    "\U0001FA70-\U0001FAFF"  # More emoji additions 🩷
    "\U0001FAB0-\U0001FABF"  # More animals 🪳
    "\U0001FAC0-\U0001FACF"  # More people 🫂
    "\U0001FAD0-\U0001FADF"  # More food 🍿
    "]+", flags=re.UNICODE
)

# -------------------------------
# Initialize Sentiment Analyzer
# -------------------------------
analyzer = SentimentIntensityAnalyzer()

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
def load_tweets(csv_file):
    """
    Load tweets from a CSV file.
    Expects CSV to have at least 'id' and 'text' columns.
    """
    return pd.read_csv(csv_file)

# -------------------------------
# Sentence and Chunk Tokenization
# -------------------------------
def split_into_sentences(text):
    """Split tweet text into sentences using NLTK."""
    return sent_tokenize(text)

def split_sentence_into_chunks(sentence):
    """
    Split a sentence into three roughly equal word chunks.
    For very short sentences, return the sentence as a single chunk.
    """
    tokens = word_tokenize(sentence)
    n = len(tokens)
    if n < 3:
        return [sentence]
    chunk_size = n // 3
    return [
        " ".join(tokens[:chunk_size]),
        " ".join(tokens[chunk_size:2*chunk_size]),
        " ".join(tokens[2*chunk_size:])
    ]

# -------------------------------
# Emoji Detection and Placement
# -------------------------------
def detect_emojis_in_text(text):
    """
    Find all emojis in the text.

    Returns:
        List of tuples: (emoji, start_index)
    """
    return [(match.group(), match.start()) for match in emoji_pattern.finditer(text)]

def determine_emoji_placement(chunks):
    """
    For each chunk, check and label every emoji found based on its relative position
    within the chunk. For each emoji in the chunk, determine its placement:
    - "start" if it appears in the first third,
    - "middle" if it appears in the middle third,
    - "end" if it appears in the final third.

    Returns:
        A list of lists; each inner list contains the labels for all emojis in that chunk.
        If no emoji is found in a chunk, returns ["none"].
    """
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
    """Remove all emojis from the text using caching to optimize repeated calls."""
    return emoji.replace_emoji(text, replace='')

# -------------------------------
# Sentiment Analysis
# -------------------------------
def get_sentiment_scores(text):
    """Compute sentiment scores using VADER."""
    return analyzer.polarity_scores(text)

def analyze_sentiment(text):
    """
    Analyze sentiment for text with and without emoji.

    Returns:
        Tuple: (sentiment_with, sentiment_without)
    """
    score_with = get_sentiment_scores(text)
    score_without = get_sentiment_scores(remove_emojis(text))
    return score_with, score_without

# -------------------------------
# Emotion Detection
# -------------------------------
def get_dominant_emotion(text):
    """
    Determine the dominant emotion in text using text2emotion.

    Returns:
        The emotion with the highest score, or None if not detected.
    """
    emotions = te.get_emotion(text)
    if emotions:
        return max(emotions, key=emotions.get)
    return None

# -------------------------------
# Processing Individual Tweets
# -------------------------------
def process_tweet(tweet_id, tweet_text):
    """
    Process a tweet by:
      - Splitting into sentences and chunks.
      - Performing sentiment and emotion analysis for each sentence with and without emoji.
      - Analyzing each chunk for emoji placement (detecting multiple emojis),
        sentiment (with/without emoji), and dominant emotion (with/without emoji).

    Returns:
        A dictionary with detailed results.
    """
    sentences = split_into_sentences(tweet_text)
    tweet_result = {"tweet_id": tweet_id, "sentences": []}

    for sentence in sentences:
        sentence_dict = {"sentence": sentence, "chunks": []}
        # Sentence-level analysis:
        sent_sentiment_with, sent_sentiment_without = analyze_sentiment(sentence)
        sent_emotion_with = get_dominant_emotion(sentence)
        sent_emotion_without = get_dominant_emotion(remove_emojis(sentence))
        sentence_dict["sentiment_with_emoji"] = sent_sentiment_with
        sentence_dict["sentiment_without_emoji"] = sent_sentiment_without
        sentence_dict["dominant_emotion_with_emoji"] = sent_emotion_with
        sentence_dict["dominant_emotion_without_emoji"] = sent_emotion_without

        chunks = split_sentence_into_chunks(sentence)
        # Get a list of emoji placements for each chunk. Each element is a list of labels.
        placements = determine_emoji_placement(chunks)

        for i, chunk in enumerate(chunks):
            sentiment_with, sentiment_without = analyze_sentiment(chunk)
            emotion_with = get_dominant_emotion(chunk)
            emotion_without = get_dominant_emotion(remove_emojis(chunk))

            chunk_dict = {
                "chunk_text": chunk,
                "emoji_placements": placements[i],  # now a list of labels (or ["none"])
                "sentiment_with_emoji": sentiment_with,
                "sentiment_without_emoji": sentiment_without,
                "dominant_emotion_with_emoji": emotion_with,
                "dominant_emotion_without_emoji": emotion_without
            }
            sentence_dict["chunks"].append(chunk_dict)
        tweet_result["sentences"].append(sentence_dict)

    return tweet_result

# -------------------------------
# Aggregating Additional Research Insights
# -------------------------------
def aggregate_insights(results):
    """
    Compute overall insights across all tweets:
      - Frequency of emoji placements.
      - Average sentiment compound score differences.
      - Distribution of dominant emotions (with and without emoji).

    Returns:
        A dictionary of aggregated insights.
    """
    placement_counter = Counter()
    emotion_counter_with = Counter()
    emotion_counter_without = Counter()
    sentiment_differences = []
    total_chunks = 0

    for tweet in results:
        for sentence in tweet["sentences"]:
            for chunk in sentence["chunks"]:
                # Count each placement label (if the chunk has multiple, count them all)
                for label in chunk["emoji_placements"]:
                    placement_counter[label] += 1
                comp_with = chunk["sentiment_with_emoji"].get("compound", 0)
                comp_without = chunk["sentiment_without_emoji"].get("compound", 0)
                sentiment_differences.append(comp_with - comp_without)
                total_chunks += 1
                if chunk["dominant_emotion_with_emoji"]:
                    emotion_counter_with[chunk["dominant_emotion_with_emoji"]] += 1
                if chunk["dominant_emotion_without_emoji"]:
                    emotion_counter_without[chunk["dominant_emotion_without_emoji"]] += 1

    avg_sentiment_diff = sum(sentiment_differences) / total_chunks if total_chunks > 0 else 0

    insights = {
        "total_tweets_processed": len(results),
        "total_chunks_processed": total_chunks,
        "emoji_placement_distribution": dict(placement_counter),
        "average_sentiment_compound_difference": avg_sentiment_diff,
        "dominant_emotion_distribution_with_emoji": dict(emotion_counter_with),
        "dominant_emotion_distribution_without_emoji": dict(emotion_counter_without)
    }
    return insights

# -------------------------------
# Main Execution Function with Parallel Processing
# -------------------------------
def main():
    # Load the tweets dataset (adjust file path as needed)
    input_file = Path("cleaned_data/all_cleaned_tweets.csv")
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    tweets_df = load_tweets(input_file)

    # Verify required columns exist
    if 'text' not in tweets_df.columns:
        raise ValueError("CSV file must contain at least a 'text' column")

    # Create tweet IDs if they don't exist
    if 'id' not in tweets_df.columns:
        tweets_df['id'] = range(1, len(tweets_df) + 1)

    # Prepare a list of (id, text) tuples for processing
    tweets_to_process = list(zip(tweets_df['id'], tweets_df['text']))
    results = []

    # Determine optimal workers (leave 3 CPU free)
    max_workers = max(1, os.cpu_count() - 3) if hasattr(os, 'cpu_count') else 4

    # Process tweets in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_tweet, tweet_id, tweet_text): (tweet_id, tweet_text)
                   for tweet_id, tweet_text in tweets_to_process}

        for future in as_completed(futures):
            tweet_id, tweet_text = futures[future]
            try:
                tweet_result = future.result()
                results.append(tweet_result)
            except Exception as exc:
                print(f"Tweet {tweet_id} processing generated an exception: {exc}")

    # Ensure output directory exists
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write detailed analysis results to JSON
    with open(output_dir / "tweet_analysis_results.json", "w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=2, ensure_ascii=False)

    # Compute and save aggregated research insights
    insights = aggregate_insights(results)
    with open(output_dir / "analysis_insights.json", "w", encoding="utf-8") as outfile:
        json.dump(insights, outfile, indent=2, ensure_ascii=False)

    print(f"Analysis complete. Processed {len(results)} tweets.")
    print(f"Detailed results saved to {output_dir/'tweet_analysis_results.json'}")
    print(f"Aggregated insights saved to {output_dir/'analysis_insights.json'}")

if __name__ == '__main__':
    main()
