from backend.app.ml_models.sentiment import predict_sentiment
from backend.app.services.summarizer_service import summarize_review

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# ─────────── Load DialoGPT Model ───────────
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ─────────── Fallback Reply (Plain DialoGPT) ───────────
def fallback_dialo_gpt_reply(user_message: str):
    input_ids = tokenizer.encode(user_message + tokenizer.eos_token, return_tensors='pt').to(device)
    chat_history_ids = model.generate(
        input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    reply = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return reply.strip()

# ─────────── Main Chatbot Logic ───────────
def generate_chatbot_reply(user_message: str):
    user_message_lower = user_message.lower()

    # Summarization
    if user_message_lower.startswith("summarize:"):
        review_text = user_message[len("summarize:"):].strip()
        if not review_text:
            return "Please provide the review text after 'summarize:'."
        return summarize_review(review_text)

    # Sentiment
    if user_message_lower.startswith("sentiment:"):
        review_text = user_message[len("sentiment:"):].strip()
        if not review_text:
            return "Please provide the review text after 'sentiment:'."
        sentiment = predict_sentiment([review_text])[0]
        return f"This review is likely **{sentiment}**."

    # Small Talk Fallback
    return fallback_dialo_gpt_reply(user_message)
