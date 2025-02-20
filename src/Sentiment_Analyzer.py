import os
import re
import torch
import pandas as pd
import emoji
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pre-trained emotion analysis model and move it to GPU
MODEL_NAME = "cardiffnlp/twitter-roberta-base-emotion"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()  # Set model to evaluation mode

# Initialize VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Define an emoji regex pattern
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

# Function to extract emojis and their positions
def extract_emojis(text):
    emojis = emoji_pattern.findall(text)
    positions = [match.start() for match in emoji_pattern.finditer(text)]
    return emojis, positions

# Function to determine emoji placement
def determine_placement(text, positions):
    if not positions:
        return 'None'
    length = len(text)
    placement = []
    for pos in positions:
        if pos == 0:
            placement.append('Start')
        elif pos == length - 1:
            placement.append('End')
        else:
            placement.append('Middle')
    return ', '.join(set(placement))  # Remove duplicates

# Function to analyze sentiment
def analyze_sentiment(text):
    return analyzer.polarity_scores(text)['compound']

# Function to analyze sentiment including emojis
def analyze_sentiment_with_emojis(text, emojis):
    text_with_emojis = text + ' ' + ''.join(emojis)  # Append emojis to text
    return analyzer.polarity_scores(text_with_emojis)['compound']

# Function to analyze emotion using pre-trained model with GPU support
def analyze_emotion(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)[0]
    labels = ["anger", "joy", "optimism", "sadness"]  # Model-specific labels
    return labels[scores.argmax().item()]

# Function to analyze emotion including emojis
def analyze_emotion_with_emojis(text, emojis):
    text_with_emojis = text + ' ' + ''.join(emojis)  # Append emojis to text
    return analyze_emotion(text_with_emojis)

# Folder containing CSV files
folder_path = "archive"
output_folder = "processed_tweets"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process all CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # Process only CSV files
        csv_path = os.path.join(folder_path, file_name)

        print(f"Processing: {file_name}")

        tweets_data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                text = line.strip()  # Remove leading/trailing spaces
                if text and text != "Text":  # Ignore empty lines and header
                    emojis, positions = extract_emojis(text)
                    placement = determine_placement(text, positions)

                    sentiment_text = analyze_sentiment(text)
                    sentiment_with_emojis = analyze_sentiment_with_emojis(text, emojis)
                    sentiment_difference = sentiment_with_emojis - sentiment_text  # Impact of emojis

                    dominant_emotion_without_emojis = analyze_emotion(text)  # Emotion without emojis
                    dominant_emotion_with_emojis = analyze_emotion_with_emojis(text, emojis)  # Emotion with emojis

                    tweets_data.append({
                        'text': text,
                        'emojis': ''.join(emojis),
                        'placement': placement,
                        'sentiment_text': sentiment_text,
                        'sentiment_with_emojis': sentiment_with_emojis,
                        'sentiment_difference': sentiment_difference,
                        'dominant_emotion_without_emojis': dominant_emotion_without_emojis,
                        'dominant_emotion_with_emojis': dominant_emotion_with_emojis
                    })

        # Convert to DataFrame
        df = pd.DataFrame(tweets_data)

        # Save processed data
        output_file = os.path.join(output_folder, f"processed_{file_name}")
        df.to_csv(output_file, index=False, encoding='utf-8')

        print(f"Saved: {output_file}")

print("All CSV files processed successfully!")
