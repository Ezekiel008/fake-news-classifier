import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import string
import os
from scipy.sparse import hstack
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="Fake News Classifier", layout="centered")

# Load models and tools with caching
@st.cache_resource
def load_tools():
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(BASE_DIR, "models")

        rf_model = joblib.load(os.path.join(model_dir, "fake_news_rf_model.pkl"))
        tfidf_vectorizer = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
        label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "meta_scaler.pkl"))

        return rf_model, tfidf_vectorizer, label_encoder, scaler

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


# ✅ CALL THE FUNCTION HERE
rf_model, tfidf_vectorizer, label_encoder, scaler = load_tools()

# Text cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text


# Category descriptions
CATEGORY_DESCRIPTIONS = {
    "bias": "Biased news, showing partiality or unfair representation.",
    "bs": "Fake or misleading information.",
    "conspiracy": "Articles promoting conspiracy theories.",
    "fake": "Fully fabricated content.",
    "hate": "Hateful, discriminatory content.",
    "junksci": "Junk science – misrepresenting scientific facts.",
    "satire": "Satirical or humorous content, not meant to mislead.",
    "state": "State-sponsored propaganda."
}

# Title and instructions
st.title("📰 Fake News Category Classifier")
st.write("Paste a news article or upload a `.txt` file. Add metadata for better prediction.")

# Input method
input_method = st.radio("Select input method:", ["📝 Paste Text", "📁 Upload .txt File"])

if input_method == "📝 Paste Text":
    user_input = st.text_area("Paste your article below:")
else:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    user_input = ""
    if uploaded_file is not None:
        try:
            user_input = uploaded_file.read().decode("utf-8")
        except Exception as e:
            st.error(f"Error reading file: {e}")

# Metadata Inputs
st.markdown("### Metadata (Optional)")
col1, col2, col3, col4, col5 = st.columns(5)

likes = col1.number_input("Likes", value=0)
comments = col2.number_input("Comments", value=0)
shares = col3.number_input("Shares", value=0)
domain_rank = col4.number_input("Domain Rank", value=1000)
published_hour = col5.slider("Published Hour", min_value=0, max_value=23, value=12)

# Predict button
if st.button("Predict Category"):

    if not user_input.strip():
        st.warning("Please provide article text.")

    elif rf_model is None:
        st.error("Model failed to load.")

    else:
        try:

            # Clean text
            cleaned = clean_text(user_input)

            # TFIDF transformation
            X_text_tfidf = tfidf_vectorizer.transform([cleaned])

            # Metadata dataframe
            meta_input = pd.DataFrame(
                [[domain_rank, likes, comments, shares, published_hour]],
                columns=['domain_rank', 'likes', 'comments', 'shares', 'published_hour']
            )

            # Scale metadata
            scaled_meta = scaler.transform(meta_input)

            # Combine features
            X_final = hstack([X_text_tfidf, scaled_meta])

            # Predict
            pred = rf_model.predict(X_final)

            pred_label = label_encoder.inverse_transform([pred[0]])[0]

            # Show result
            st.success(f"🎯 Predicted Category: **{pred_label.upper()}**")

            st.markdown(
                f"🗂 **Category Description:** {CATEGORY_DESCRIPTIONS.get(pred_label, 'No description available.')}"
            )

            # Prediction probabilities
            proba = rf_model.predict_proba(X_final)[0]

            st.markdown("### 📊 Prediction Probabilities")

            prob_df = pd.DataFrame({
                "Category": label_encoder.classes_,
                "Probability": proba
            }).sort_values(by="Probability", ascending=False)

            st.bar_chart(prob_df.set_index("Category"))

        except Exception as e:
            st.error(f"Prediction failed: {e}")