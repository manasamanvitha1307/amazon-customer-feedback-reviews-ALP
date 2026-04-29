from transformers import pipeline

# Load summarizer from Hugging Face (lazy load to avoid multiple downloads)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(texts, min_length=30, max_length=100):
    results = []
    for text in texts:
        if len(text.split()) < 8:
            results.append("Text too short to summarize.")
        else:
            summary = summarizer(text, min_length=min_length, max_length=max_length)
            results.append(summary[0]['summary_text'])
    return results
