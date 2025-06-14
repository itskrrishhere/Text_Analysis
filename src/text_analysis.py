import json
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# -------------------------------
# Data Loading Functions
# -------------------------------
def load_json(filepath):
    """
    Load JSON data from a file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        data: Parsed JSON data.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_data():
    """
    Load aggregated insights and detailed tweet analysis results,
    and flatten the nested tweet analysis structure into a DataFrame.

    Returns:
        insights (dict): Aggregated insights.
        df (DataFrame): Flattened DataFrame of tweet chunks.
    """
    # Load aggregated insights
    insights = load_json('output/analysis_insights.json')

    # Load detailed tweet results
    detailed_results = load_json('output/tweet_analysis_results.json')

    # Flatten the nested JSON structure into a list of dictionaries for each chunk.
    # Note: the updated tweet processing now returns "emoji_placements" (a list).
    chunk_data = []
    for tweet in detailed_results:
        for sentence in tweet['sentences']:
            for chunk in sentence['chunks']:
                chunk_data.append({
                    # Use "emoji_placements" from the updated processing; if not present, fallback to "emoji_placement".
                    "emoji_placements": chunk.get('emoji_placements', chunk.get('emoji_placement', ["none"])),
                    "sentiment_with": chunk['sentiment_with_emoji']['compound'],
                    "sentiment_without": chunk['sentiment_without_emoji']['compound'],
                    "emotion_with": chunk['dominant_emotion_with_emoji'],
                    "emotion_without": chunk['dominant_emotion_without_emoji']
                })

    # Create DataFrame and compute sentiment difference.
    df = pd.DataFrame(chunk_data)
    df['sentiment_diff'] = df['sentiment_with'] - df['sentiment_without']

    # For visualization purposes, create a string version of emoji placements.
    # If multiple placements are present, join them with a comma.
    df['emoji_placement_str'] = df['emoji_placements'].apply(lambda x: ','.join(x) if isinstance(x, list) else x)

    return insights, df

# -------------------------------
# Visualization Functions
# -------------------------------
def plot_emoji_placement_distribution(insights):
    """
    Plot a bar chart showing the distribution of emoji placements.

    Args:
        insights (dict): Aggregated insights containing emoji placement counts.
    """
    placement_counts = insights.get('emoji_placement_distribution', {})
    placements = list(placement_counts.keys())
    counts = list(placement_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(placements, counts, color='skyblue')
    plt.title('Distribution of Emoji Placements in Text Chunks')
    plt.xlabel('Emoji Placement Position')
    plt.ylabel('Number of Chunks')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_sentiment_boxplot(df):
    """
    Create a boxplot for sentiment differences by emoji placement.
    Here we use the joined string of emoji placements.

    Args:
        df (DataFrame): DataFrame with tweet chunk data.
    """
    # Exclude rows where no emoji is present.
    filtered = df[df['emoji_placement_str'] != 'none']

    # Define groups based on placement strings.
    groups = filtered.groupby('emoji_placement_str')['sentiment_diff'].apply(list)

    plt.figure(figsize=(12, 6))
    plt.boxplot(groups, labels=groups.index, patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
    plt.title('Impact of Emoji Placement on Sentiment Difference\n(With vs Without Emojis)')
    plt.xlabel('Emoji Placement(s)')
    plt.ylabel('Sentiment Score Difference')
    plt.tight_layout()
    plt.show()

def plot_emotion_comparison(insights):
    """
    Plot side-by-side bar charts comparing the dominant emotion distributions
    for text with and without emojis.

    Args:
        insights (dict): Aggregated insights containing emotion distribution data.
    """
    emotion_with = insights.get('dominant_emotion_distribution_with_emoji', {})
    emotion_without = insights.get('dominant_emotion_distribution_without_emoji', {})

    # Combine keys so both plots have the same x-axis categories.
    all_emotions = set(emotion_with.keys()) | set(emotion_without.keys())
    all_emotions = sorted(all_emotions)

    counts_with = [emotion_with.get(emotion, 0) for emotion in all_emotions]
    counts_without = [emotion_without.get(emotion, 0) for emotion in all_emotions]

    x = np.arange(len(all_emotions))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, counts_with, width, label='With Emojis', color='lightgreen')
    plt.bar(x + width/2, counts_without, width, label='Without Emojis', color='orchid')

    plt.xlabel('Emotion Category')
    plt.ylabel('Frequency')
    plt.title('Dominant Emotion Distribution: With vs Without Emojis')
    plt.xticks(x, all_emotions, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_sentiment_histogram(df):
    """
    Plot a histogram with a Gaussian kernel density overlay for sentiment score differences.

    Args:
        df (DataFrame): DataFrame containing sentiment difference data.
    """
    plt.figure(figsize=(10, 6))
    counts, bins, _ = plt.hist(df['sentiment_diff'], bins=30, density=True, alpha=0.6, color='g')

    density = stats.gaussian_kde(df['sentiment_diff'])
    xs = np.linspace(df['sentiment_diff'].min(), df['sentiment_diff'].max(), 200)
    plt.plot(xs, density(xs), 'k--', linewidth=2, label='Density')

    mean_val = df['sentiment_diff'].mean()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_val:.2f}')

    plt.title('Distribution of Sentiment Score Differences')
    plt.xlabel('Sentiment Difference (With Emojis - Without Emojis)')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------
# Statistical Analysis Functions
# -------------------------------
def perform_statistical_tests(df):
    """
    Conduct statistical tests on the sentiment difference data.
    Performs a one-sample T-test and an ANOVA test for groups defined by emoji placement.

    Args:
        df (DataFrame): DataFrame containing sentiment difference and emoji placement data.
    """
    # One-sample T-test: Test if the mean sentiment difference is statistically different from 0.
    t_stat, p_value = stats.ttest_1samp(df['sentiment_diff'], 0)
    print("One-Sample T-Test for Sentiment Difference:")
    print(f"  T-statistic = {t_stat:.2f}")
    print(f"  p-value     = {p_value:.4f}\n")

    # ANOVA: Test if the mean sentiment differences vary across emoji placement groups (excluding 'none').
    filtered = df[df['emoji_placement_str'] != 'none']
    groups = [group['sentiment_diff'].values for name, group in filtered.groupby('emoji_placement_str')]

    if len(groups) > 1:
        f_stat, p_value_anova = stats.f_oneway(*groups)
        print("ANOVA Test for Sentiment Difference Across Emoji Placements:")
        print(f"  F-statistic = {f_stat:.2f}")
        print(f"  p-value     = {p_value_anova:.4f}\n")
    else:
        print("Not enough groups for ANOVA test (need at least two emoji placement groups).\n")

def print_additional_insights(df):
    """
    Calculate and print additional insights in a tabular format.

    Args:
        df (DataFrame): DataFrame containing tweet chunk data.
    """
    # Consider only rows where emoji is present.
    emoji_present = df[df['emoji_placement_str'] != 'none']

    avg_sent_with = emoji_present['sentiment_with'].mean()
    avg_sent_without = emoji_present['sentiment_without'].mean()
    avg_sent_diff = emoji_present['sentiment_diff'].mean()
    # Frequency of emotion change: proportion of rows where the dominant emotion changes when emojis are removed.
    emotion_change_freq = (df['emotion_with'] != df['emotion_without']).mean()

    insight_table = pd.DataFrame({
        "Metric": [
            "Avg Sentiment (With Emojis)",
            "Avg Sentiment (Without Emojis)",
            "Avg Sentiment Change",
            "Emotion Change Frequency"
        ],
        "Value": [
            avg_sent_with,
            avg_sent_without,
            avg_sent_diff,
            emotion_change_freq
        ]
    })

    print("Additional Insights:")
    print(insight_table.to_string(index=False))

# -------------------------------
# Main Execution Function
# -------------------------------
def main():
    # Load data and aggregated insights
    insights, df = load_data()

    # Generate visualizations
    plot_emoji_placement_distribution(insights)
    plot_sentiment_boxplot(df)
    plot_emotion_comparison(insights)
    plot_sentiment_histogram(df)

    # Perform statistical tests and display results
    perform_statistical_tests(df)

    # Print additional insights
    print_additional_insights(df)

if __name__ == '__main__':
    main()
