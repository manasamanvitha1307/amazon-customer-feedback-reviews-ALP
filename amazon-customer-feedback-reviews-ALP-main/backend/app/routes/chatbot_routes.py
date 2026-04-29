from fastapi import APIRouter, Request
from pydantic import BaseModel
from backend.app.services.chatbot_service import generate_chatbot_reply

router = APIRouter()

class ChatInput(BaseModel):
    message: str

@router.post("/chatbot")
def chatbot_endpoint(input: ChatInput):
    reply = generate_chatbot_reply(input.message)  # ✅ Fixed: only one argument
    return {"reply": reply}

