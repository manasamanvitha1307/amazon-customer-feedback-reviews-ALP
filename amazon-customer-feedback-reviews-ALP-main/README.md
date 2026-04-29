#  Customer Feedback Analysis using NLP

Transform customer feedback into real-time, actionable insights using Natural Language Processing (NLP).  
This project helps businesses analyze sentiment, detect fake reviews, and make smarter product decisions automatically.


##  Project Overview

This system:
- Understands customer emotions and intentions using **DistilBERT**
- Detects fake reviews with **Logistic Regression**
- Works in real time with a simple **Streamlit** dashboard
- Helps product teams, customer support, and analysts act faster

---

##  Problem Statement

Customer feedback is vital for improving user experience and informing business strategy.  
However, review data is often unstructured, noisy, and overwhelming—making it hard to extract meaningful insights.

This NLP pipeline solves these challenges by:
- Categorizing reviews to uncover product/service themes
- Summarizing feedback to highlight core user concerns
- Performing sentiment analysis to gauge satisfaction
- Detecting and filtering fake reviews (~88% accuracy)  
  *Distinguishes between authentic and generated content to ensure reliability*
  
The goal is to build a system that distinguishes between genuine and fake reviews with high accuracy.

---

##  Text Preprocessing Techniques

To clean and prepare the review data, we used:
- Removing punctuation characters
- Transforming text to lowercase
- Eliminating stopwords
- Stemming
- Lemmatizing
- Removing digits

---

##  Machine Learning Algorithms Used

- **Logistic Regression** – For fake review detection  
- **Linear Regression** – Applied for certain modeling tasks as part of exploratory experimentation  
- **DistilBERT** – For sentiment and emotion classification

---

##  Features

- **Sentiment Prediction** – Classifies feedback as positive, negative, or neutral
- **Emotion Classifier** – Detects emotional tone (e.g., joy, anger, sadness)
- **Fake Review Detection** – Identifies potentially fraudulent reviews (~88% accuracy)
- **Conversational Chatbot** – Interacts with users to assist with feedback interpretation and queries

---

##  Technologies Used

- **Python** (backend & frontend)
- **FastAPI** (backend API)
- **Streamlit** (frontend UI)
- **Transformers, spaCy, scikit-learn, NLTK** (NLP)
- **Docker & Docker Compose** (DB containerization)
- **PostgreSQL** (database)

---

## 📁 Project Structure

```
.
├── backend/             # FastAPI backend
├── frontend/            # Streamlit frontend
├── docker-compose.yml   # Database setup
├── requirements.txt     # Root-level dependencies (if any)
```

---

##  Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/manojvuddanti/amazon-customer-feedback-reviews-ALP.git
cd amazon-customer-feedback-reviews-ALP
```

---

### 2. Set Up Backend

```bash
cd backend

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install any missing packages if needed
```

---

### 3. Set Up Frontend

```bash
cd ../frontend

# Make sure you're still in the virtual environment

# Install dependencies
pip install -r requirements.txt

# (Optional) Install any additional packages if required
```

---

### 4. Start PostgreSQL with Docker

From the **project root**, run:

```bash
docker compose up --build
```

This builds and starts the database container.

---

### 5. Start the Backend Server

From the root or `backend/` directory:

```bash
uvicorn backend.main:app --reload
```

---

### 6. Start the Frontend App

From the root or `frontend/` directory:

```bash
streamlit run frontend/streamlit_app.py
```

---

## Login Credentials

Use these to log in on the frontend:

- **Username**: `manoj`
- **Password**: `Man@DSP123`

## Contributing

Found a bug or have ideas to improve?  
Feel free to [open an issue](https://github.com/manojvuddanti/amazon-customer-feedback-reviews-ALP/issues) or submit a pull request.

---

## License

This project is licensed under [MIT License](LICENSE) – use it freely.
