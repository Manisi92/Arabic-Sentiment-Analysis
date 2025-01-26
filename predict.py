from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load the pre-trained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('arabic_sentiment_model')
model = BertForSequenceClassification.from_pretrained('arabic_sentiment_model')


def predict_sentiment(text):
    # Tokenize the input text with explicit max_length
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    # Debugging: print logits to understand model's output
    print("Logits:", logits)

    # Determine the predicted class (highest logit value)
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Map the predicted class to a sentiment label
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    return sentiment_map.get(predicted_class, "Unknown")  # Default to "Unknown" if class is not in the map


if __name__ == "__main__":
    texts = [
        "أنت رائع",  # Aspettativa: Positive
        "الشمس مشرقة",  # Aspettativa: Neutral
        "اليوم سيء جدا"  # Aspettativa: Negative
    ]

    for text in texts:
        print(f"Text: {text} -> Sentiment: {predict_sentiment(text)}")
