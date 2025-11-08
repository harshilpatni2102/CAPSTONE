"""
Customer Sentiment Analysis - Streamlit Web Application
This app allows users to predict sentiment of product reviews in real-time.
"""

import streamlit as st
import joblib
import json
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import preprocessing utilities
from utils.preprocessing import clean_text, get_sentiment_label

# Page configuration
st.set_page_config(
    page_title="Customer Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with unique gradient theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    }
    
    .block-container p, .block-container label, .block-container span {
        color: #333 !important;
    }
    
    .stTextArea textarea {
        font-size: 16px;
        border-radius: 15px;
        border: 2px solid #667eea;
        padding: 15px;
        background: #f8f9ff;
        color: #333 !important;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #764ba2;
        box-shadow: 0 0 20px rgba(118, 75, 162, 0.3);
        background: white;
    }
    
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: white;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    h1, h2, h3 {
        color: #764ba2 !important;
        font-weight: 700;
    }
    
    /* Fix main content text colors */
    .main p, .main span, .main label, .main div:not([class*="result-box"]):not([class*="metric-card"]) {
        color: #333 !important;
    }
    
    .result-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        animation: fadeIn 0.5s ease-in;
    }
    
    .positive-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .negative-box {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
    }
    
    .neutral-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: scale(0.9); }
        to { opacity: 1; transform: scale(1); }
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
    }
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    .stAlert {
        border-radius: 15px;
        border-left: 5px solid #667eea;
    }
    
    /* Ensure all text in main content area is dark */
    .stMarkdown, .stText {
        color: #333 !important;
    }
    
    /* Fix placeholder text */
    .stTextArea textarea::placeholder {
        color: #999 !important;
    }
    
    /* Fix warning/info text */
    .stAlert p {
        color: #333 !important;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models and vectorizer."""
    models_dir = Path(__file__).parent.parent / 'models'
    
    try:
        # Load best model
        best_model = joblib.load(models_dir / 'best_model.pkl')
        
        # Load vectorizer
        vectorizer = joblib.load(models_dir / 'tfidf_vectorizer.pkl')
        
        # Load metadata
        with open(models_dir / 'model_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Try to load all models
        all_models = {}
        model_files = {
            'Logistic Regression': 'logistic_regression_model.pkl',
            'Naive Bayes': 'naive_bayes_model.pkl',
            'SVM': 'svm_model.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = models_dir / filename
            if model_path.exists():
                all_models[model_name] = joblib.load(model_path)
        
        return best_model, vectorizer, metadata, all_models
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please run the Jupyter notebook first to train and save the models.")
        return None, None, None, None


def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for given text with intelligent neutral detection."""
    # Clean text
    cleaned_text = clean_text(text)
    
    if not cleaned_text.strip():
        return None, None, None
    
    # Vectorize
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    sentiment_label = get_sentiment_label(prediction)
    
    # Get probabilities if available
    probabilities = None
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(text_vectorized)[0]
    elif hasattr(model, 'decision_function'):
        # For SVM without probability=True, use decision_function
        # and convert to probability-like scores using softmax
        decision_scores = model.decision_function(text_vectorized)[0]
        
        # Apply softmax to convert decision scores to probabilities
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probabilities = exp_scores / exp_scores.sum()
    
    # INTELLIGENT NEUTRAL DETECTION
    if probabilities is not None and len(probabilities) == 3:
        pos_prob = probabilities[2]  # Positive probability
        neg_prob = probabilities[0]  # Negative probability
        neu_prob = probabilities[1]  # Neutral probability
        
        # Define keywords for mixed sentiment detection
        positive_keywords = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'nice', 'awesome', 'fantastic', 'wonderful']
        negative_keywords = ['bad', 'poor', 'terrible', 'worst', 'hate', 'awful', 'horrible', 'disappointing', 'useless', 'waste']
        contrast_words = ['but', 'however', 'though', 'although', 'except', 'yet', 'while']
        
        # Count sentiment indicators in the text
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)
        has_contrast = any(word in text_lower for word in contrast_words)
        
        # Override prediction to Neutral if mixed sentiment detected
        if (pos_count > 0 and neg_count > 0) or \
           (has_contrast and pos_count > 0 and neg_count > 0) or \
           (abs(pos_prob - neg_prob) < 0.3 and max(pos_prob, neg_prob) < 0.7):
            # Mixed sentiment: has both positive and negative aspects
            prediction = 1  # Set to Neutral
            sentiment_label = 'Neutral'
            # Adjust probabilities to reflect neutrality
            total = pos_prob + neg_prob
            if total > 0:
                neu_prob = 0.5 + (min(pos_prob, neg_prob) / total) * 0.3
                pos_prob = pos_prob * (1 - neu_prob) / total
                neg_prob = neg_prob * (1 - neu_prob) / total
                probabilities = np.array([neg_prob, neu_prob, pos_prob])
    
    return prediction, sentiment_label, probabilities


def get_sentiment_color(sentiment):
    """Get color for sentiment."""
    colors = {
        'Positive': '#28a745',
        'Neutral': '#ffc107',
        'Negative': '#dc3545'
    }
    return colors.get(sentiment, '#6c757d')


def get_sentiment_emoji(sentiment):
    """Get emoji for sentiment."""
    emojis = {
        'Positive': '😊',
        'Neutral': '😐',
        'Negative': '😞'
    }
    return emojis.get(sentiment, '🤔')


def main():
    """Main application function."""
    
    # Load models
    best_model, vectorizer, metadata, all_models = load_models()
    
    if best_model is None:
        st.stop()
    
    # Header
    st.title("🎯 Customer Sentiment Analysis")
    st.markdown("### Analyze sentiment of product reviews in real-time")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h1 style='text-align: center; color: white;'>📊 Model Information</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.header("🤖 Model Selection")
        available_models = list(all_models.keys()) if all_models else [metadata.get('best_model', 'Best Model')]
        
        selected_model_name = st.selectbox(
            "Choose a model:",
            available_models,
            index=0,
            help="Select a machine learning model for sentiment prediction"
        )
        
        # Use selected model
        selected_model = all_models.get(selected_model_name, best_model) if all_models else best_model
        
        st.markdown("---")
        
        # Model performance
        if metadata and 'model_performance' in metadata:
            st.header("� Model Stats")
            perf_df = pd.DataFrame(metadata['model_performance'])
            
            for _, row in perf_df.iterrows():
                if row['Model'] == selected_model_name:
                    col_m1, col_m2 = st.columns(2)
                    with col_m1:
                        st.metric("Accuracy", f"{row['Accuracy']:.1%}")
                        st.metric("Precision", f"{row['Precision']:.1%}")
                    with col_m2:
                        st.metric("Recall", f"{row['Recall']:.1%}")
                        st.metric("F1-Score", f"{row['F1-Score']:.1%}")
                    break
        
        st.markdown("---")
        
        # Dataset info
        if metadata:
            st.header("📁 Dataset Info")
            st.info(f"**Training:** {metadata.get('training_samples', 'N/A')} reviews")
            st.info(f"**Testing:** {metadata.get('test_samples', 'N/A')} reviews")
            st.info(f"**Features:** {metadata.get('feature_count', 'N/A')} TF-IDF")
        
        st.markdown("---")
        
        # About
        st.header("ℹ️ About")
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>
        This application uses Machine Learning to analyze the sentiment of customer product reviews.
        
        <br><br><b>Sentiment Classes:</b>
        <br>😊 Positive
        <br>😐 Neutral
        <br>😞 Negative
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<h2 style='color: #667eea;'>📝 Enter Review Text</h2>", unsafe_allow_html=True)
        
        # Text input
        review_text = st.text_area(
            "Enter Review",
            height=200,
            placeholder="Example: This phone is amazing! Great camera quality and long battery life.",
            help="Type or paste a product review to analyze its sentiment",
            label_visibility="collapsed"
        )
        
        # Predict button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            predict_button = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("<h2 style='color: #667eea;'>🎯 Results</h2>", unsafe_allow_html=True)
        
        result_placeholder = st.empty()
        
        # Display instructions if no prediction yet
        if not predict_button and not review_text:
            with result_placeholder.container():
                st.markdown("""
                    <div style='background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); 
                                padding: 3rem; border-radius: 20px; text-align: center; border: 2px dashed #667eea;'>
                        <h3 style='color: #667eea;'>👈 Ready to Analyze</h3>
                        <p style='color: #666; font-size: 16px;'>Enter a review and click the button to see results</p>
                    </div>
                """, unsafe_allow_html=True)
    
    # Prediction
    if predict_button and review_text:
        if len(review_text.strip()) < 10:
            st.warning("⚠️ Please enter a longer review (at least 10 characters)")
        else:
            with st.spinner("Analyzing sentiment..."):
                prediction, sentiment_label, probabilities = predict_sentiment(
                    review_text, selected_model, vectorizer
                )
                
                if sentiment_label is None:
                    st.error("❌ Could not analyze the review. Please try again with different text.")
                else:
                    # Display results
                    with col2:
                        with result_placeholder.container():
                            # Sentiment result
                            emoji = get_sentiment_emoji(sentiment_label)
                            
                            # Determine box class
                            box_class = "positive-box" if sentiment_label == "Positive" else "negative-box" if sentiment_label == "Negative" else "neutral-box"
                            
                            st.markdown(
                                f"<div class='result-box {box_class}'>"
                                f"<h1 style='color: white; font-size: 72px; margin: 0;'>{emoji}</h1>"
                                f"<h2 style='color: white; margin: 10px 0; font-size: 36px;'>{sentiment_label}</h2>"
                                f"<p style='color: rgba(255,255,255,0.9); font-size: 16px;'>Sentiment Detected</p>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                            
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Confidence scores
                            if probabilities is not None and len(probabilities) == 3:
                                st.subheader("📊 Confidence Distribution")
                                
                                labels = ['Negative', 'Neutral', 'Positive']
                                colors_list = ['#ee0979', '#f093fb', '#38ef7d']
                                
                                # Create enhanced bar chart
                                fig = go.Figure(data=[
                                    go.Bar(
                                        x=labels,
                                        y=probabilities * 100,
                                        marker=dict(
                                            color=colors_list,
                                            line=dict(color='rgba(255,255,255,0.5)', width=2)
                                        ),
                                        text=[f"{p:.1f}%" for p in probabilities * 100],
                                        textposition='outside',
                                        textfont=dict(size=14, color='#333'),
                                    )
                                ])
                                
                                fig.update_layout(
                                    title=dict(
                                        text="Probability Distribution",
                                        font=dict(size=18, color='#667eea')
                                    ),
                                    xaxis_title="Sentiment Category",
                                    yaxis_title="Confidence (%)",
                                    yaxis_range=[0, 105],
                                    showlegend=False,
                                    height=350,
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(102,126,234,0.05)',
                                    font=dict(size=12, color='#333'),
                                    margin=dict(t=60, b=40, l=40, r=40)
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Display confidence metrics in cards
                                st.markdown("<br>", unsafe_allow_html=True)
                                for i, (label, prob) in enumerate(zip(labels, probabilities)):
                                    col_metric = st.columns([1])[0]
                                    with col_metric:
                                        st.markdown(
                                            f"<div class='metric-card'>"
                                            f"<h4 style='color: {colors_list[i]}; margin: 0;'>{label}</h4>"
                                            f"<p style='font-size: 24px; font-weight: 700; color: #333; margin: 5px 0;'>{prob*100:.2f}%</p>"
                                            f"</div>",
                                            unsafe_allow_html=True
                                        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%); 
                    border-radius: 20px; margin-top: 2rem;'>
            <h3 style='color: #667eea; margin-bottom: 10px;'>Customer Sentiment Analysis</h3>
            <p style='color: #666; font-size: 14px; margin: 5px 0;'>
                Built by <b>Anshul Jadwani, Harshil Patni & Dhruv Hirani</b>
            </p>
            <p style='color: #888; font-size: 12px; margin: 5px 0;'>
                NMIMS University | BTech in AI & Data Science
            </p>
            <p style='color: #999; font-size: 11px; margin-top: 10px;'>
                Powered by Support Vector Machine with 94.69% accuracy | Trained on 3,316 e-commerce reviews
            </p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
