import os
import joblib
from .text_utils import text_process  # matches training reference

# 📦 Load the logistic regression model
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "model", "logistic_regression_pipeline.pkl"))
model = joblib.load(MODEL_PATH)

# 🔮 Predict function
def predict_fake_reviews(texts):
    return model.predict(texts)