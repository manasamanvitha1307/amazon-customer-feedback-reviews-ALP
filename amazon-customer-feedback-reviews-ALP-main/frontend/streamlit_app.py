import streamlit as st
import requests
import json
import pandas as pd
import os

# ───────────────────────────────────────────────
# 🔐 Authentication Check
# ───────────────────────────────────────────────

if "token" not in st.session_state:
    if "mode" not in st.session_state:
        st.session_state.mode = "login"

    st.set_page_config(page_title="🔐 Amazon Dashboard login", layout="centered")
    st.title("🔐 Amazon Customer Review Analysis")

    logo_path = os.path.join(os.path.dirname(__file__), "assests", "logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=100)
    else:
        st.warning("⚠️ Logo image not found in assests folder.")

    if st.session_state.mode == "login":
        st.markdown("Don't have an account? click on this 👇 ", unsafe_allow_html=True)
        if st.button("Create Account"):
            st.session_state.mode = "register"

    if st.session_state.mode == "register":
        st.markdown("Already registered? 🔙", unsafe_allow_html=True)
        if st.button("Back to Login"):
            st.session_state.mode = "login"

    # --- LOGIN VIEW ---
    if st.session_state.mode == "login":
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_submit = st.form_submit_button("Login")

        if login_submit:
            payload = {"username": username, "password": password}
            try:
                with st.spinner("🔐 Verifying credentials..."):
                    res = requests.post("http://localhost:8000/login", json=payload)
                    if res.status_code == 200:
                        st.session_state["token"] = res.json()["access_token"]
                        st.session_state["username"] = username
                        st.success(f"✅ Welcome, {username}!")
                        st.rerun()
                    else:
                        st.write("Response Text:", res.text)
                        st.error("❌ Invalid credentials.")
            except Exception as e:
                st.error(f"🚨 Login failed: {str(e)}")

    elif st.session_state.mode == "register":
        with st.form("register_form"):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            register_submit = st.form_submit_button("Register")

        if register_submit:
            if new_password != confirm_password:
                st.error("⚠️ Passwords do not match.")
            elif len(new_password) < 6:
                st.warning("📏 Password too short (min 6 chars).")
            else:
                payload = {"username": new_username, "password": new_password}
                try:
                    with st.spinner("🛠️ Creating account..."):
                        res = requests.post("http://localhost:8000/register", json=payload)
                        if res.status_code == 200:
                            st.success("🎉 User registered successfully! You can now login.")
                            st.session_state.mode = "login"
                        else:
                            st.error("❌ Registration failed. Try again.")
                except Exception as e:
                    st.error(f"🚨 Error: {str(e)}")

# ───────────────────────────────────────────────
# 🧠 Logged-in Dashboard
# ───────────────────────────────────────────────

else:
    st.set_page_config(page_title="Amazon Dashboard", layout="wide")

    # Sidebar
    st.sidebar.title("📊 Features")
    selected_feature = st.sidebar.radio("Choose a Feature to analyze:", [
        "Sentiment Analysis",
        "Emotion Detection",
        "Review Summarization",
        "Fake Review Detection"
    ])

    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

    # ─── Chat Toggle Button ───
    if "show_chat" not in st.session_state:
        st.session_state["show_chat"] = False

    for _ in range(18):
        st.sidebar.write("")
    if st.sidebar.button("👤 Chat Assistant"):
        st.session_state["show_chat"] = not st.session_state["show_chat"]

    st.title("🛒 Amazon Feedback Dashboard")
    st.success(f"🔓 Logged in as `{st.session_state['username']}`")

    headers = {"Authorization": f"Bearer {st.session_state['token']}"}

    # Common inputs
    text = ""
    df = None

    if selected_feature in ["Sentiment Analysis", "Review Summarization"]:
        text = st.text_area("Enter a review")
        uploaded_file = st.file_uploader("Upload a JSONL file", type=["jsonl"])

        if uploaded_file:
            raw_data = [json.loads(line.decode("utf-8")) for line in uploaded_file]
            df = pd.DataFrame(raw_data)
            st.write("📄 Uploaded Data Sample:")
            st.write(df.head())

    # --- Sentiment Analysis ---
    if selected_feature == "Sentiment Analysis":
        st.header("🧠 Sentiment Analysis")

        if st.button("Analyze Review"):
            if text:
                res = requests.post("http://localhost:8000/predict_sentiment", json={"texts": [text]}, headers=headers)
                if res.status_code == 200:
                    sentiment = res.json()['predictions'][0]
                    st.success(f"🧠 Sentiment: {sentiment}")
                else:
                    st.error("API error during prediction.")
            else:
                st.warning("Enter a review first.")
                
        if st.button("Batch Sentiment Analysis"):
            if df is not None:
                import sys
                import os
                # Dynamically add ../backend to Python path
                sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
                #st.write("📁 sys.path includes:", sys.path)
                from app.ml_models.sentiment import predict_sentiment
                review_col = df.columns[0]
                import random
                sampled_df = df.sample(n=10, random_state=random.randint(0, 9999))
                texts = sampled_df[review_col].astype(str).tolist()
                sentiments = predict_sentiment(texts)
                sampled_df["sentiment"] = sentiments
                st.success("✅ Sentiment analysis complete for 10 random reviews!")
                st.write(sampled_df[[review_col, "sentiment"]].head(10))
                # Prepare data
                st.session_state["pending_predictions"] = sampled_df

            else:
                st.warning("Upload a file first.")

        if st.button("save to database"):
            try:
                df_display = st.session_state.get("pending_predictions")

                if df_display is not None and not df_display.empty:
                    review_column = df_display.columns[0]  # Safely grab it again here

                    payload = {
                        "sentiments": df_display[[review_column, "sentiment"]]
                            .rename(columns={review_column: "review"})
                            .to_dict(orient="records")
                    }

                    res = requests.post("http://localhost:8000/save_sentiments", json=payload)

                    if res.ok:
                        st.success("💾 Sentiment batch saved to database!")
                    else:
                        st.error(f"❌ Save failed: {res.status_code}")
                else:
                    st.warning("⚠️ No sentiment data available to save.")
            except Exception as e:
                        st.error(f"❌ Failed to process file: {e}")


    # --- Emotion Detection ---
    elif selected_feature == "Emotion Detection":
        st.header("🎭 Emotion Detection")
        input_text = st.text_area("Paste a review to detect emotion")
        EMOJI_MAP = {
            "joy": "😊", "sadness": "😢", "anger": "😠",
            "fear": "😨", "surprise": "😲", "disgust": "🤢", "neutral": "😐"
        }
        if input_text and st.button("Detect Emotion"):
            response = requests.post("http://localhost:8000/detect_emotion", json={"texts": [input_text]}, headers=headers)
            if response.status_code == 200:
                emotion = response.json()['emotions'][0]
                symbol = EMOJI_MAP.get(emotion, "❔")
                st.success(f"🎭 Emotion: {emotion} {symbol}")
            else:
                st.error(f"API Error: {response.status_code}")

    # --- Review Summarization ---
    elif selected_feature == "Review Summarization":
        st.header("📝 Review Summarization")
        if st.button("Summarize Review") and text.strip():
            response = requests.post("http://localhost:8000/summarize_review", json={"texts": [text]}, headers=headers)
            if response.status_code == 200:
                summary = response.json()['summaries'][0]
                st.subheader("🔍 Summary")
                st.success(summary)
            else:
                st.error("API error during summarization.")
        elif text.strip() == "":
            st.warning("Enter a review to summarize.")

    # --- Fake Review Detection ---
    elif selected_feature == "Fake Review Detection":
        st.header("🕵️‍♂️ Fake Review Detection")
        input_text = st.text_area("Paste a review to check for authenticity")
        uploaded_file = st.file_uploader("Upload a CSV with reviews", type=["csv"])

        if input_text and st.button("Check Review"):
            try:
                payload = {"text": input_text}
                res = requests.post("http://localhost:8000/predict_fake_review", json=payload, headers=headers)
                if res.status_code == 200:
                    result = res.json()
                    pred = result["fake"]
                    score = result["confidence_score"]
                    if pred == 1:
                        st.error(f"🧢 Fake Review (Confidence: {score}%)")
                    else:
                        st.success(f"✅ Genuine Review (Confidence: {100 - score}%)")
                else:
                    st.error(f"API error: {res.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")

        if uploaded_file and st.button("Run Batch Detection"):
            try:
                df = pd.read_csv(uploaded_file)
                reviews = df["text_"].astype(str).tolist()
                res = requests.post("http://localhost:8000/predict_fake_review_batch", json={"texts": reviews}, headers=headers)
                if res.status_code == 200:
                    results = res.json()["results"]
                    df_display = pd.DataFrame(results)
                    st.session_state["pending_predictions"] = df_display
                    st.success(f"✅ Processed {len(results)} reviews")
                    st.dataframe(df_display[["review", "verdict"]])
                    st.download_button("📥 Download CSV", data=df_display.to_csv(index=False).encode("utf-8"), file_name="results.csv")
                else:
                    st.error(f"API error: {res.status_code}")
            except Exception as e:
                st.error(f"Batch detection failed: {e}")

    
    # ─────────────────────────────────────────────
    # 💬 Chat Panel (Only visible when toggled)
    # ─────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if st.session_state["show_chat"]:
        st.markdown("---")
        st.subheader("👤 Chat Assistant")
        st.caption("- Type: `summarize: your review text` to get a summary of a review.")
        st.caption("- Type: `sentiment: your review text` to analyze the review's sentiment.")
        st.caption("Ask anything related to reviews, features, or this dashboard.")

        user_input = st.chat_input("Ask your question...")

        if user_input:
            st.session_state["chat_history"].append({"role": "user", "text": user_input})

            try:
                headers = {"Authorization": f"Bearer {st.session_state['token']}"}
                response = requests.post(
                    "http://localhost:8000/chatbot",
                    json={"message": user_input},
                    headers=headers
                )

                if response.status_code == 200:
                    bot_reply = response.json()["reply"]
                else:
                    bot_reply = f"⚠️ Error from server: {response.status_code}"

            except Exception as e:
                bot_reply = f"❌ Failed to connect: {e}"

            st.session_state["chat_history"].append({"role": "bot", "text": bot_reply})

        for msg in st.session_state["chat_history"]:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["text"])
            else:
                st.chat_message("assistant").write(msg["text"])
