// app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import re
from textblob import TextBlob
import joblib
import warnings
import os # <-- IMPORT OS MODULE FOR BETTER PATH HANDLING (Optional but good practice)

warnings.filterwarnings('ignore')

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# Page configuration
st.set_page_config(
    page_title="Group 1 Spam Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern Professional CSS (Omitted for brevity, assuming this section is unchanged)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --bg-primary: #0A0B0D;
        --bg-secondary: #111213;
        --bg-elevated: #1E1F23;
        --text-primary: #ECECED;
        --text-secondary: #9B9C9E;
        --border-color: #26272B;
        --accent-primary: #3B82F6;
        --success: #10B981;
        --error: #EF4444;
        --font-family: 'Inter', sans-serif;
    }
    
    * { font-family: var(--font-family); }
    .stApp { background: var(--bg-primary); color: var(--text-primary); }
    
    .app-header {
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
    }
    
    .app-title {
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .app-subtitle {
        font-size: 0.95rem;
        color: var(--text-secondary);
    }
    
    .stButton > button {
        background: var(--accent-primary);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 500;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: #2563EB;
        transform: translateY(-1px);
    }
    
    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid;
    }
    
    .result-spam {
        background: rgba(239, 68, 68, 0.1);
        border-color: var(--error);
    }
    
    .result-ham {
        background: rgba(16, 185, 129, 0.1);
        border-color: var(--success);
    }
    
    .result-title {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .result-spam .result-title { color: var(--error); }
    .result-ham .result-title { color: var(--success); }
    
    .result-confidence {
        font-size: 2.5rem;
        font-weight: 700;
        line-height: 1;
        margin-bottom: 0.25rem;
    }
    
    .metric-card {
        background: var(--bg-elevated);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-primary);
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: var(--text-secondary);
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state (Unchanged)
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "Manual SVM"
if 'history' not in st.session_state:
    st.session_state.history = []
if 'total_scanned' not in st.session_state:
    st.session_state.total_scanned = 0
if 'spam_detected' not in st.session_state:
    st.session_state.spam_detected = 0
if 'current_message' not in st.session_state:
    st.session_state.current_message = ""

# Download NLTK resources (Unchanged)
@st.cache_resource
def download_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger',
                 'vader_lexicon', 'punkt_tab', 'omw-1.4']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass

download_nltk_resources()

# Initialize NLP tools (Unchanged)
@st.cache_resource
def init_nlp_tools():
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    sia = SentimentIntensityAnalyzer()
    return stop_words, lemmatizer, sia

STOP_WORDS, LEMMATIZER, SIA = init_nlp_tools()

# --- 🎯 THE CRITICAL FIX IS HERE ---
# Load models
@st.cache_resource
def load_all_models():
    """Load all available models and provide explicit error reporting."""
    models = {}
    
    # Define model configurations
    model_configs = {
        'Logistic Regression': 'models/logistic_regression_spam_detector_model.pkl',
        'Naive Bayes': 'models/naive_bayes_spam_detector.pkl',
        'Manual SVM': 'models/svm_spam_detector.pkl'
    }

    # Use a dictionary to track missing models for a clear report
    missing_models = []
    
    # Attempt to load each model
    for name, path in model_configs.items():
        try:
            # Use os.path.join for better cross-platform compatibility
            full_path = os.path.join(os.path.dirname(__file__), path)
            models[name] = joblib.load(full_path)
            # st.success(f"Loaded model: {name}") # You can uncomment this for successful load check
        except FileNotFoundError:
            missing_models.append(f"Model file not found: {path}")
        except Exception as e:
            missing_models.append(f"Error loading {name} from {path}: {e}")

    # If models are missing, display the detailed error in the app
    if missing_models:
        st.error("🚨 Critical Error: Failed to load one or more models.")
        for error_detail in missing_models:
             st.code(error_detail, language='text')
        
    return models

# Text preprocessing (Unchanged)
def advanced_text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    text = re.sub(r'\d+', 'NUM', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOP_WORDS]
    return tokens

# Feature extraction (Unchanged)
def extract_features(text):
    processed = " ".join(advanced_text_preprocessing(text))
    features = {
        'message': text,
        'processed_message': processed,
        'message_length': len(text),
        'word_count': len(text.split()),
        'char_count': len(text.replace(" ", "")),
        'avg_word_length': len(text.replace(" ", "")) / (len(text.split()) + 1),
        'punctuation_count': len(re.findall(r'[^\w\s]', text)),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_count': sum(1 for c in text if c.isupper()),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1),
        'has_url': 1 if re.search(r'http|www', text.lower()) else 0,
        'has_email': 1 if re.search(r'\S+@\S+', text) else 0,
        'digit_count': sum(c.isdigit() for c in text),
        'digit_ratio': sum(c.isdigit() for c in text) / (len(text) + 1),
        'textblob_sentiment': TextBlob(text).sentiment.polarity,
        'textblob_subjectivity': TextBlob(text).sentiment.subjectivity,
        'vader_compound': SIA.polarity_scores(text)['compound'],
        'vader_pos': SIA.polarity_scores(text)['pos'],
        'vader_neg': SIA.polarity_scores(text)['neg'],
        'spam_word_count': sum(1 for word in ['free', 'win', 'winner', 'cash',
                                              'prize', 'claim', 'call', 'urgent', 'txt']
                               if word in text.lower())
    }
    return features

# Prediction (Unchanged)
def predict_message(text, model):
    if model is None:
        return None, None, None
    features = extract_features(text)
    df_pred = pd.DataFrame([features])
    prediction = model.predict(df_pred)[0]
    # Handle models without predict_proba (e.g., a custom SVM)
    try:
        probability = model.predict_proba(df_pred)[0]
    except AttributeError:
        # Fallback: Assign a dummy probability based on prediction for display purposes
        if prediction == 'spam':
            probability = np.array([0.0, 1.0])
        else:
            probability = np.array([1.0, 0.0])
            
    return prediction, probability, features

# Create gauge chart (Unchanged)
def create_gauge(value, color_scheme):
    color = "#EF4444" if color_scheme == "danger" else "#10B981"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'suffix': "%", 'font': {'size': 40, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#3A3B40"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "#1A1B1E",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': "#111213"},
                {'range': [50, 100], 'color': "#1E1F23"}
            ]
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#ECECED"},
        height=250,
        margin=dict(l=10, r=10, t=30, b=10)
    )
    return fig

# Header (Unchanged)
st.markdown("""
<div class="app-header">
    <div class="app-title">🛡️ Group 1 Spam Detector</div>
    <div class="app-subtitle">AI-powered message analysis with machine learning (LR, NB, Manual SVM)</div>
</div>
""", unsafe_allow_html=True)

# Load all models
all_models = load_all_models()

if not all_models:
    st.error("⚠️ No model files found and all loading attempts failed. Please ensure file paths and contents are correct.")
    st.stop()

# Remaining Streamlit UI/Logic (Unchanged)
# ... (rest of the app.py file)