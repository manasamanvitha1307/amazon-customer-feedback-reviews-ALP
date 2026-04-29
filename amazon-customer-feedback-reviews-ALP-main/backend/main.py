from pydantic import BaseModel
from fastapi import APIRouter
from fastapi import BackgroundTasks
from typing import List
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from backend.app.ml_models.sentiment import predict_sentiment
from backend.auth import verify_password, create_access_token, hash_password, verify_token
from backend.auth import SECRET_KEY, ALGORITHM
from backend.app.ml_models.summarizer import summarize_text
from backend.app.ml_models.emotion import predict_emotion
from fastapi import FastAPI, Form, HTTPException, Depends
from pydantic import BaseModel
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import pandas as pd
import io
import joblib
import asyncio
import asyncpg
from datetime import datetime


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

app = FastAPI()

try:
    pipeline = joblib.load("backend/model/fake_review_pipeline.joblib")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {str(e)}")


class ReviewInput(BaseModel):
    texts: List[str]

class SingleReviewInput(BaseModel):
    text: str

# Temporary user store (use DB later)
#users_db = {"manoj": hash_password("Man@DSP123")}

@app.get("/health")
def health_check():
    return {"status": "OK"}

from fastapi import FastAPI, Depends, HTTPException
from backend.app.database.databse import SessionLocal, engine
from backend.model.user import User
from backend.app.database.schemas import UserCreate, Token
from backend.auth import hash_password, verify_password, create_access_token, verify_token

from sqlalchemy.orm import Session

# Create DB tables
from backend.app.database.databse import Base
Base.metadata.create_all(bind=engine)


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already exists.")
    
    new_user = User(
        username=user.username,
        hashed_password=hash_password(user.password)
    )
    db.add(new_user)
    db.commit()
    return {"message": "User created"}

@app.post("/login", response_model=Token)
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": user.username})
    return {"access_token": token}

@app.get("/secure-data")
def secure_data(username: str = Depends(verify_token)):
    return {"message": f"Welcome {username}, you're authenticated!"}

@app.post("/predict_sentiment")
def predict(input: ReviewInput, user: str = Depends(verify_token)):
    return {"user": user, "predictions": predict_sentiment(input.texts)}

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=403, detail="Invalid token payload")
        return username
    except JWTError:
        raise HTTPException(status_code=403, detail="Token verification failed")


@app.post("/summarize_review")
def summarize(input: ReviewInput):
    return {"summaries": summarize_text(input.texts)}

@app.post("/detect_emotion")
def detect_emotion(payload: ReviewInput):
    emotions = predict_emotion(payload.texts)
    return {"emotions": emotions}

@app.post("/predict_fake_review")
def predict_fake_review(review: SingleReviewInput):
    try:
        if not review.text or review.text.strip() == "":
            raise HTTPException(status_code=400, detail="Empty review text.")

        prediction = pipeline.predict([review.text])[0]
        probability = pipeline.predict_proba([review.text])[0][1]
        score = int(probability * 100)

        return {
            "input": review.text,
            "fake": bool(prediction),
            "confidence_score": score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Create pool at startup
@app.on_event("startup")
async def startup():
    app.state.db_pool = await asyncpg.create_pool(dsn="postgresql://postgres:team_alpha@localhost:5050/feedbackdb")

# Close pool on shutdown
@app.on_event("shutdown")
async def shutdown():
    await app.state.db_pool.close()

async def save_to_postgres_async(results: List[dict]):
    try:
        pool = app.state.db_pool
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO review_predictions (review, label, confidence_score, verdict, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                    """,
                    [
                        (
                            sanitize(r["review"]),
                            r["label"],
                            r["confidence_score"],
                            r["verdict"],
                            datetime.utcnow()
                        ) for r in results
                    ]
                )
    except Exception as e:
        print(f"❌ DB error: {e}")

# ✅ Sync wrapper to run async DB logger
def sync_save(results: List[dict]):
    asyncio.run(save_to_postgres_async(results))

def sanitize(text):
    return text.strip().replace("\n", " ")[:1000]  # cap length


@app.post("/predict_fake_review_batch")
def predict_fake_review_batch(batch: ReviewInput):
    import random

    max_batch = 500
    process_limit = 10

    incoming_reviews = batch.texts[:max_batch]
    reviews = random.sample(incoming_reviews, min(len(incoming_reviews), process_limit))

    if not reviews:
        raise HTTPException(status_code=400, detail="No reviews provided.")

    predictions = pipeline.predict(reviews)
    probabilities = pipeline.predict_proba(reviews)[:, 1]
    scores = [int(p * 100) for p in probabilities]

    results = []
    for text, pred, score in zip(reviews, predictions, scores):
        verdict = "🧢 Fake" if pred else "✅ Genuine"
        results.append({
            "review": text,
            "label": bool(pred),
            "confidence_score": score,
            "verdict": f"{verdict} ({score}%)"
        })
    return {"results": results}

@app.post("/save_predictions")
def save_predictions(payload: dict, background_tasks: BackgroundTasks):
    results = payload.get("predictions")

    if not results:
        raise HTTPException(status_code=400, detail="No predictions provided.")

    background_tasks.add_task(save_to_postgres_async, results)  # ✅ Use directly, no asyncio.run
    return {"message": "✅ Saved successfully."}

class SentimentItem(BaseModel):
    review: str
    sentiment: str

# 🧼 Simple sanitizer
def sanitize(text: str) -> str:
    return text.strip().replace("\n", " ")[:1000]

# 🛠️ Async saver task
async def save_sentiment_batch(items: List[dict]):
    try:
        pool = app.state.db_pool
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(
                    """
                    INSERT INTO review_sentiments (review, sentiment, created_at)
                    VALUES ($1, $2, $3)
                    """,
                    [
                        (
                            sanitize(item["review"]),
                            item["sentiment"],
                            datetime.utcnow()
                        )
                        for item in items
                    ]
                )
        print(f"✅ Inserted {len(items)} sentiment records.")
    except Exception as e:
        print(f"❌ DB error during sentiment save: {e}")

# 🚀 Endpoint to trigger save
@app.post("/save_sentiments")
def save_sentiments(payload: dict, background_tasks: BackgroundTasks):
    sentiments = payload.get("sentiments")
    if not sentiments:
        raise HTTPException(status_code=400, detail="No sentiment data provided.")
    background_tasks.add_task(save_sentiment_batch, sentiments)
    return {"message": "✅ Sentiments batch saved."}

from backend.app.routes import chatbot_routes
app.include_router(chatbot_routes.router)
