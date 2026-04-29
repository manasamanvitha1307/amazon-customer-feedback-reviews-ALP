from fastapi import APIRouter
from pydantic import BaseModel
from app.services.inference import predict_sentiment_distilbert

router = APIRouter()

class Review(BaseModel):
    text: str

@router.post("/distilbert")
def get_sentiment_distilbert(review: Review):
    return predict_sentiment_distilbert(review.text)
