# Text_Analysis

This project analyzes the syntax and semantics of emojis used in tweets, including their placement, frequency, and impact on sentiment and emotion.

## Folder Structure

- `archive.zip` — Compressed raw tweet CSV datasets.
- `archive/` — Unzipped CSV files for each emoji/tweet set.
- `cleaned_data/` — Output folder for cleaned tweets.
- `output/` — Output folder for processed results and figures.
- `src/text_cleaning.py` — Script for cleaning and preprocessing raw tweets.
- `src/text_processing.py` — Script for analyzing cleaned tweets (sentiment, emoji placement, emotion).
- `src/text_analyis.py` — Script for visualizing and statistically analyzing results.
- `src/` — Source code folder.

## Setup

1. **Install dependencies** (Python 3.8+ recommended):

    ```sh
    pip install pandas matplotlib seaborn nltk emoji vaderSentiment text2emotion tqdm
    ```

2. **Unzip the data**:

    ```sh
    cd archive
    unzip ../archive.zip
    cd ..
    ```

## Usage

1. **Clean the tweets**  
   Run the cleaning script to preprocess the raw data:

    ```sh
    python src/text_cleaning.py
    ```

2. **Analyze the cleaned tweets**  
   Run the processing script to analyze emoji placement, sentiment, and emotion:

    ```sh
    python src/text_processing.py
    ```

3. **Visualize and analyze results**  
   Run the analysis script to generate plots and statistical summaries:

    ```sh
    python src/text_analyis.py
    ```

## Output

- Cleaned tweets: `cleaned_data/all_cleaned_tweets.csv`
- Detailed analysis: `output/tweet_analysis_results.jsonl`
- Aggregated insights: `output/analysis_insights.json`
- Figures: Saved in `output/` or displayed interactively

## Notes

- Ensure the `archive/` and `cleaned_data/` folders exist and contain the expected files.
- The main analysis and visualization are performed in `src/text_analyis.py`.
- For interactive exploration, you can adapt the scripts into Jupyter notebooks.


## License

This project is for educational purposes.