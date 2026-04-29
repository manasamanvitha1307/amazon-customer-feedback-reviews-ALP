from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load once during startup
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained("path_to_your_trained_model")
model.eval()

def predict_sentiment_distilbert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    label = "Positive" if prediction == 1 else "Negative"
    return {"label": label, "confidence": torch.softmax(logits, dim=1).max().item()}
