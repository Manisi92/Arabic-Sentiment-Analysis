import pandas as pd
import re
from camel_tools.tokenizers.word import simple_word_tokenize


# Function to clean the text (already fixed in previous steps)
def preprocess_text(text):
    if not isinstance(text, str):  # Handle non-string values like NaN
        return ""

    text = re.sub(r"http\S+", "", text)  # Remove URLs
    # Add any additional preprocessing steps here

    return text


# Tokenize the Arabic text
def tokenize_arabic_text(text):
    if not isinstance(text, str):  # Handle non-string values
        return []  # Return an empty list if the text is not a string
    return simple_word_tokenize(text)


# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Preprocess the text (cleaning)
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Tokenize the cleaned text
    df['tokenized_text'] = df['cleaned_text'].apply(tokenize_arabic_text)

    return df


# Example usage:
data = load_and_preprocess_data('data/data.csv')
print(data.head())