from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load summarization model (PEGASUS)
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")
model.eval()

def summarize_review(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=60,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
