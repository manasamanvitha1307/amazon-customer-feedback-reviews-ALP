import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Set device — automatically uses GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
import os

model_path = os.path.join(os.path.dirname(__file__), "..", "..", "model")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.to(device)
model.eval()

# Map model outputs to labels
labels = {0: "Negative", 1: "Positive"}

def predict_sentiment(texts, batch_size=32, max_length=128):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Tokenize and move inputs to device
        encodings = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length
        ).to(device)

        # Disable gradients for inference
        with torch.no_grad():
            outputs = model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=1).tolist()
            results.extend([labels[pred] for pred in predictions])

    return results