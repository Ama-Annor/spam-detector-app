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

# Modern Professional CSS
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

# Initialize session state
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

# Download NLTK resources
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

# Initialize NLP tools
@st.cache_resource
def init_nlp_tools():
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    sia = SentimentIntensityAnalyzer()
    return stop_words, lemmatizer, sia

STOP_WORDS, LEMMATIZER, SIA = init_nlp_tools()

# Load models
@st.cache_resource
def load_all_models():
    """Load all available models"""
    models = {}
    
    # Try to load Logistic Regression
    try:
        models['Logistic Regression'] = joblib.load('models/logistic_regression_spam_detector_model.pkl')
    except:
        pass
    
    # Try to load Naive Bayes
    try:
        models['Naive Bayes'] = joblib.load('models/naive_bayes_spam_detector.pkl')
    except:
        pass
    
    # Try to load Manual SVM
    try:
        models['Manual SVM'] = joblib.load('models/svm_spam_detector.pkl')
    except:
        pass
    
    return models

# Text preprocessing
def advanced_text_preprocessing(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', 'URL', text)
    text = re.sub(r'\S+@\S+', 'EMAIL', text)
    text = re.sub(r'\d+', 'NUM', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOP_WORDS]
    return tokens

# Feature extraction
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

# Prediction
def predict_message(text, model):
    if model is None:
        return None, None, None
    features = extract_features(text)
    df_pred = pd.DataFrame([features])
    prediction = model.predict(df_pred)[0]
    probability = model.predict_proba(df_pred)[0]
    return prediction, probability, features

# Create gauge chart
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

# Header
st.markdown("""
<div class="app-header">
    <div class="app-title">🛡️ Group 1 Spam Detector</div>
    <div class="app-subtitle">AI-powered message analysis with machine learning (LR, NB, Manual SVM)</div>
</div>
""", unsafe_allow_html=True)

# Load all models
all_models = load_all_models()

if not all_models:
    st.error("⚠️ No model files found. Please ensure model files are in the directory.")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("", ["Detector", "Analytics", "History", "About"],
                    label_visibility="collapsed")
    
    st.divider()
    
    # Model selector
    st.markdown("### Model Selection")
    available_models = list(all_models.keys())
    selected_model_name = st.selectbox(
        "Choose Model",
        available_models,
        index=available_models.index(
            st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
        key="model_selector"
    )
    
    if selected_model_name != st.session_state.selected_model:
        st.session_state.selected_model = selected_model_name
        st.rerun()
    
    model = all_models[st.session_state.selected_model]
    st.caption(f"Active: {st.session_state.selected_model}")
    
    st.divider()
    
    # Statistics
    st.markdown("### Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Scanned", st.session_state.total_scanned)
    with col2:
        st.metric("Spam", st.session_state.spam_detected)
    
    if st.session_state.total_scanned > 0:
        spam_rate = (st.session_state.spam_detected / st.session_state.total_scanned) * 100
        st.metric("Spam Rate", f"{spam_rate:.1f}%")
    
    st.divider()
    if st.button("Clear Data", use_container_width=True, type="secondary"):
        st.session_state.history = []
        st.session_state.total_scanned = 0
        st.session_state.spam_detected = 0
        st.session_state.current_message = ""
        st.rerun()

# Main Content
if page == "Detector":
    st.markdown("### Analyze Message")
    
    # Example buttons
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🎁 Spam Example", use_container_width=True, type="secondary", key="spam_ex"):
            st.session_state.current_message = "Congratulations! You've won $1000! Click here now!"
            st.rerun()
    with col2:
        if st.button("✅ Legit Example", use_container_width=True, type="secondary", key="legit_ex"):
            st.session_state.current_message = "Hey, are you free for lunch tomorrow?"
            st.rerun()
    with col3:
        if st.button("⚠️ Phishing Example", use_container_width=True, type="secondary", key="phish_ex"):
            st.session_state.current_message = "URGENT! Your account has been suspended. Verify now: bit.ly/secure"
            st.rerun()
    with col4:
        if st.button("💼 Business Example", use_container_width=True, type="secondary", key="biz_ex"):
            st.session_state.current_message = "Meeting at 3 PM. Please bring the reports."
            st.rerun()
    
    # Text area
    message_input = st.text_area(
        "Enter message to analyze",
        value=st.session_state.current_message,
        height=150,
        placeholder="Type or paste your message here...",
        key="message_textarea"
    )
    
    st.markdown("")
    if st.button("🚀 Analyze Message", use_container_width=True, type="primary"):
        if message_input:
            st.session_state.current_message = message_input
            
            with st.spinner("Analyzing..."):
                progress = st.progress(0)
                for i in range(100):
                    time.sleep(0.005)
                    progress.progress(i + 1)
                
                prediction, probability, features = predict_message(message_input, model)
                
                st.session_state.total_scanned += 1
                if prediction == 'spam':
                    st.session_state.spam_detected += 1
                
                st.session_state.history.insert(0, {
                    'timestamp': datetime.now(),
                    'message': message_input[:100] + "..." if len(message_input) > 100 else message_input,
                    'prediction': prediction,
                    'confidence': max(probability) * 100,
                    'model': st.session_state.selected_model
                })
                st.session_state.history = st.session_state.history[:100]
            
            st.divider()
            st.markdown("### Results")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if prediction == 'spam':
                    st.markdown(f"""
                    <div class="result-card result-spam">
                        <div class="result-title">🚫 Spam Detected</div>
                        <div class="result-confidence">{max(probability)*100:.1f}%</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card result-ham">
                        <div class="result-title">✅ Legitimate Message</div>
                        <div class="result-confidence">{max(probability)*100:.1f}%</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.plotly_chart(
                    create_gauge(
                        probability[1] if prediction == 'spam' else probability[0],
                        "danger" if prediction == 'spam' else "success"
                    ),
                    use_container_width=True
                )
            
            st.markdown("### Probabilities")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{probability[0]*100:.2f}%</div>
                    <div class="metric-label">Legitimate</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{probability[1]*100:.2f}%</div>
                    <div class="metric-label">Spam</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### Message Details")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{features['message_length']}</div>
                    <div class="metric-label">Characters</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{features['word_count']}</div>
                    <div class="metric-label">Words</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{features['exclamation_count']}</div>
                    <div class="metric-label">Exclamations</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{features['spam_word_count']}</div>
                    <div class="metric-label">Spam Words</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a message to analyze")

elif page == "Analytics":
    st.markdown("### Analytics Dashboard")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        col1, col2, col3, col4 = st.columns(4)
        spam_count = (df['prediction'] == 'spam').sum()
        ham_count = (df['prediction'] == 'ham').sum()
        
        with col1:
            st.metric("Total Messages", len(df))
        with col2:
            st.metric("Spam", spam_count)
        with col3:
            st.metric("Legitimate", ham_count)
        with col4:
            st.metric("Avg Confidence", f"{df['confidence'].mean():.1f}%")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=[spam_count, ham_count], names=['Spam', 'Legitimate'],
                        title="Distribution", color_discrete_sequence=['#EF4444', '#10B981'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#ECECED"}, height=350)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(df, x='confidence', nbins=20, title="Confidence Distribution",
                             color_discrete_sequence=['#3B82F6'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1B1E',
                            font={'color': "#ECECED"}, height=350)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available. Start analyzing messages to see analytics.")

elif page == "History":
    st.markdown("### Analysis History")
    
    if st.session_state.history:
        filter_type = st.selectbox("Filter", ["All", "Spam Only", "Legitimate Only"])
        
        history = st.session_state.history.copy()
        if filter_type == "Spam Only":
            history = [h for h in history if h['prediction'] == 'spam']
        elif filter_type == "Legitimate Only":
            history = [h for h in history if h['prediction'] == 'ham']
        
        st.info(f"Showing {len(history)} messages")
        
        for idx, entry in enumerate(history[:20]):
            emoji = '🚫' if entry['prediction'] == 'spam' else '✅'
            pred = entry['prediction'].upper()
            model_used = entry.get('model', 'Unknown')
            with st.expander(f"{emoji} {entry['timestamp'].strftime('%Y-%m-%d %H:%M')} - {pred} ({entry['confidence']:.1f}%) - {model_used}"):
                st.text_area("Message", entry['message'], height=100, disabled=True,
                           key=f"hist_{idx}", label_visibility="collapsed")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Confidence", f"{entry['confidence']:.1f}%")
                with col2:
                    st.metric("Model", model_used)
    else:
        st.info("No history available. Start analyzing messages!")

else:  # About
    st.markdown("### About")
    
    st.markdown("""
    <div class="metric-card">
        <h4>Group 1 Spam Detector</h4>
        <p>An AI-powered spam detection system featuring three machine learning algorithms:</p>
        <ul>
            <li><strong>Logistic Regression</strong> - Binary classification baseline</li>
            <li><strong>Naive Bayes</strong> - Probabilistic text classifier</li>
            <li><strong>Manual SVM</strong> - From-scratch Support Vector Machine implementation</li>
        </ul>
        <p>The system analyzes over 20 linguistic features to accurately classify messages.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### Model Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Logistic Regression</div>
            <div class="metric-value">98.6%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Naive Bayes</div>
            <div class="metric-value">98.2%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Manual SVM</div>
            <div class="metric-value">96.9%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #6B6C6F; font-size: 0.875rem; padding: 1.5rem 0;">
    Group 1 Spam Detector © 2025 | Built with Streamlit
</div>
""", unsafe_allow_html=True)
