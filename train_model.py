import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load the preprocessed data
data = pd.read_csv(r'C:\Users\andre\arabic-sentiment-analysis\data\data.csv')

# Check for missing values in the dataset
print("Missing values in dataset:")
print(data.isnull().sum())

# Ensure 'label' column exists, create it if it does not
if 'label' not in data.columns:
    def label_text(text):
        # Basic example: label as positive (2), neutral (1), or negative (0) based on keywords
        if isinstance(text, str):  # Ensure the text is a string
            if "positive" in text.lower():
                return 2  # Positive sentiment
            elif "neutral" in text.lower():
                return 1  # Neutral sentiment
        return 0  # Negative sentiment (default)


    data['label'] = data['text'].apply(label_text)

# Drop rows where 'text' or 'label' columns have NaN values
data = data.dropna(subset=['text', 'label'])

# Ensure the 'label' column is of integer type
data['label'] = data['label'].astype(int)
data['text'] = data['text'].astype(str)

# Verify the first few rows
print("First few rows with labels:")
print(data[['text', 'label']].head())

# Use the 'text' column for features and 'label' column for target
X = data['text']  # 'text' column contains the text data
Y = data['label']  # 'label' column contains the sentiment labels

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Load pre-trained AraBERT tokenizer
tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')


def tokenize_function(examples):
    # Tokenizing the text and ensuring padding/truncation
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)


# Convert data to Hugging Face Dataset format
train_dataset = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
train_dataset = train_dataset.map(tokenize_function, batched=True)

test_dataset = Dataset.from_dict({'text': X_test.tolist(), 'label': y_test.tolist()})
test_dataset = test_dataset.map(tokenize_function, batched=True)

# Load the model
model = BertForSequenceClassification.from_pretrained('asafaya/bert-base-arabic',
                                                      num_labels=3)  # Assuming 3 labels: positive, neutral, negative

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',  # Output directory
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    evaluation_strategy="epoch",  # Evaluation strategy
    logging_dir='./logs',  # Directory for storing logs
    logging_steps=10,  # Log every 10 steps
)

# Define Trainer
trainer = Trainer(
    model=model,  # The model to train
    args=training_args,  # Training arguments
    train_dataset=train_dataset,  # Training dataset
    eval_dataset=test_dataset  # Evaluation dataset
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('arabic_sentiment_model')
tokenizer.save_pretrained('arabic_sentiment_model')