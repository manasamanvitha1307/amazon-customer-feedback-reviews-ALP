import joblib

# Load the saved pipeline
pipeline = joblib.load("backend/model/fake_review_pipeline.joblib")

# Predict a sample review
text = "love well made sturdi comfort i love veri pretti"
prediction = pipeline.predict([text])[0]
prob = pipeline.predict_proba([text])[0][1]
print("Fake review?", bool(prediction))
score = int(pipeline.predict_proba([text])[0][1] * 100)
print(f"⚠️ Fake Review Detected with {score}% Confidence!")