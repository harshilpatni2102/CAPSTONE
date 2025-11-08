"""
Customer Sentiment Analysis - Main Pipeline Script
This script orchestrates the entire sentiment analysis pipeline.
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Import utilities
from utils.preprocessing import clean_text, convert_sentiment_to_numeric

# Import ML libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)


class SentimentAnalysisPipeline:
    """Main pipeline for sentiment analysis."""
    
    def __init__(self, data_path, models_dir='models'):
        """
        Initialize the pipeline.
        
        Parameters:
        -----------
        data_path : str
            Path to the CSV data file
        models_dir : str
            Directory to save trained models
        """
        self.data_path = data_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.models = {}
        self.results = {}
    
    def load_data(self):
        """Load and perform initial data exploration."""
        print("📂 Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"✅ Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        print(f"\n📊 Sentiment distribution:")
        print(self.df['Sentiment'].value_counts())
        
        return self
    
    def preprocess_data(self):
        """Preprocess the data."""
        print("\n🧹 Preprocessing data...")
        
        # Handle missing values
        self.df['title'] = self.df['title'].fillna('')
        self.df['body'] = self.df['body'].fillna('')
        self.df['mobile_names'] = self.df['mobile_names'].fillna('Unknown')
        self.df = self.df.dropna(subset=['Sentiment'])
        
        # Combine title and body
        self.df['review_text'] = self.df['title'] + ' ' + self.df['body']
        
        # Convert sentiment to numeric
        self.df['sentiment_numeric'] = self.df['Sentiment'].apply(convert_sentiment_to_numeric)
        
        # Clean text
        print("   Cleaning text (this may take a few minutes)...")
        self.df['cleaned_text'] = self.df['review_text'].apply(
            lambda x: clean_text(x, remove_stopwords_flag=True, lemmatize=True)
        )
        
        # Remove empty reviews
        self.df = self.df[self.df['cleaned_text'].str.strip() != '']
        self.df = self.df.reset_index(drop=True)
        
        print(f"✅ Preprocessing complete: {len(self.df)} reviews remaining")
        
        return self
    
    def prepare_features(self, test_size=0.2):
        """Prepare features for training."""
        print(f"\n🔧 Preparing features (test size: {test_size})...")
        
        X = self.df['cleaned_text']
        y = self.df['sentiment_numeric']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # TF-IDF Vectorization
        print("   Applying TF-IDF vectorization...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        self.X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        self.X_test_tfidf = self.vectorizer.transform(self.X_test)
        
        print(f"✅ Features prepared:")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Test samples: {len(self.X_test)}")
        print(f"   Feature dimensions: {self.X_train_tfidf.shape[1]}")
        
        return self
    
    def train_models(self):
        """Train all models."""
        print("\n🤖 Training models...")
        
        # Logistic Regression
        print("\n   Training Logistic Regression...")
        lr_model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        lr_model.fit(self.X_train_tfidf, self.y_train)
        self.models['Logistic Regression'] = lr_model
        self._evaluate_model('Logistic Regression', lr_model)
        
        # Naive Bayes
        print("\n   Training Naive Bayes...")
        nb_model = MultinomialNB()
        nb_model.fit(self.X_train_tfidf, self.y_train)
        self.models['Naive Bayes'] = nb_model
        self._evaluate_model('Naive Bayes', nb_model)
        
        # SVM
        print("\n   Training SVM...")
        svm_model = SVC(kernel='linear', probability=True, random_state=42)
        svm_model.fit(self.X_train_tfidf, self.y_train)
        self.models['SVM'] = svm_model
        self._evaluate_model('SVM', svm_model)
        
        print("\n✅ All models trained successfully!")
        
        return self
    
    def _evaluate_model(self, model_name, model):
        """Evaluate a single model."""
        y_pred = model.predict(self.X_test_tfidf)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred
        }
        
        print(f"      Accuracy: {accuracy:.4f}")
        print(f"      Precision: {precision:.4f}")
        print(f"      Recall: {recall:.4f}")
        print(f"      F1-Score: {f1:.4f}")
    
    def compare_models(self):
        """Compare all trained models."""
        print("\n📊 Model Comparison:")
        print("=" * 80)
        print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("=" * 80)
        
        for model_name, metrics in self.results.items():
            print(f"{model_name:<25} "
                  f"{metrics['accuracy']:<12.4f} "
                  f"{metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} "
                  f"{metrics['f1_score']:<12.4f}")
        
        print("=" * 80)
        
        # Find best model
        best_model_name = max(self.results, key=lambda x: self.results[x]['accuracy'])
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   Accuracy: {self.results[best_model_name]['accuracy']:.4f}")
        
        return best_model_name
    
    def save_models(self, best_model_name):
        """Save all models and artifacts."""
        print(f"\n💾 Saving models to {self.models_dir}...")
        
        # Save best model
        joblib.dump(self.models[best_model_name], self.models_dir / 'best_model.pkl')
        print(f"   ✅ Best model saved: best_model.pkl")
        
        # Save all models
        for model_name, model in self.models.items():
            filename = model_name.lower().replace(' ', '_') + '_model.pkl'
            joblib.dump(model, self.models_dir / filename)
            print(f"   ✅ {model_name} saved: {filename}")
        
        # Save vectorizer
        joblib.dump(self.vectorizer, self.models_dir / 'tfidf_vectorizer.pkl')
        print(f"   ✅ Vectorizer saved: tfidf_vectorizer.pkl")
        
        # Save metadata
        metadata = {
            'best_model': best_model_name,
            'model_performance': [
                {
                    'Model': name,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1-Score': metrics['f1_score']
                }
                for name, metrics in self.results.items()
            ],
            'feature_count': self.X_train_tfidf.shape[1],
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test)
        }
        
        with open(self.models_dir / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"   ✅ Metadata saved: model_metadata.json")
        
        print("\n✅ All models and artifacts saved successfully!")
        
        return self
    
    def run_pipeline(self):
        """Run the complete pipeline."""
        print("\n" + "=" * 80)
        print("🚀 STARTING SENTIMENT ANALYSIS PIPELINE")
        print("=" * 80)
        
        self.load_data()
        self.preprocess_data()
        self.prepare_features()
        self.train_models()
        best_model_name = self.compare_models()
        self.save_models(best_model_name)
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        return self


def predict_sentiment(review_text, models_dir='models'):
    """
    Predict sentiment for a given review with intelligent neutral detection.
    
    Parameters:
    -----------
    review_text : str
        The review text to analyze
    models_dir : str
        Directory containing trained models
    """
    models_dir = Path(models_dir)
    
    # Load model and vectorizer
    model = joblib.load(models_dir / 'best_model.pkl')
    vectorizer = joblib.load(models_dir / 'tfidf_vectorizer.pkl')
    
    # Clean and predict
    cleaned = clean_text(review_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    
    sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(vectorized)[0]
        
        # INTELLIGENT NEUTRAL DETECTION
        # Check for mixed sentiment indicators
        pos_prob = probabilities[2]  # Positive probability
        neg_prob = probabilities[0]  # Negative probability
        neu_prob = probabilities[1]  # Neutral probability
        
        # Define keywords for mixed sentiment detection
        positive_keywords = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'perfect', 'nice', 'awesome']
        negative_keywords = ['bad', 'poor', 'terrible', 'worst', 'hate', 'awful', 'horrible', 'disappointing']
        contrast_words = ['but', 'however', 'though', 'although', 'except', 'yet']
        
        # Count sentiment indicators in the text
        text_lower = review_text.lower()
        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)
        has_contrast = any(word in text_lower for word in contrast_words)
        
        # Override prediction to Neutral if mixed sentiment detected
        if (pos_count > 0 and neg_count > 0) or \
           (has_contrast and pos_count > 0 and neg_count > 0) or \
           (abs(pos_prob - neg_prob) < 0.3 and max(pos_prob, neg_prob) < 0.7):
            # Mixed sentiment: has both positive and negative aspects
            prediction = 1  # Set to Neutral
            # Adjust probabilities to reflect neutrality
            total = pos_prob + neg_prob
            if total > 0:
                neu_prob = 0.5 + (min(pos_prob, neg_prob) / total) * 0.3
                pos_prob = pos_prob * (1 - neu_prob) / total
                neg_prob = neg_prob * (1 - neu_prob) / total
                probabilities = np.array([neg_prob, neu_prob, pos_prob])
        
        confidence = probabilities[prediction]
        
        return {
            'sentiment': sentiment_map[prediction],
            'confidence': confidence,
            'probabilities': {
                'Negative': probabilities[0],
                'Neutral': probabilities[1],
                'Positive': probabilities[2]
            }
        }
    else:
        return {
            'sentiment': sentiment_map[prediction],
            'confidence': None,
            'probabilities': None
        }


def main():
    """Main function to handle CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Customer Sentiment Analysis Pipeline'
    )
    
    parser.add_argument(
        'command',
        choices=['train', 'predict'],
        help='Command to execute: train or predict'
    )
    
    parser.add_argument(
        '--data',
        default='data/clean_review.csv',
        help='Path to the data file (for training)'
    )
    
    parser.add_argument(
        '--models-dir',
        default='models',
        help='Directory to save/load models'
    )
    
    parser.add_argument(
        '--text',
        help='Review text to predict (for prediction)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        # Run training pipeline
        pipeline = SentimentAnalysisPipeline(args.data, args.models_dir)
        pipeline.run_pipeline()
        
    elif args.command == 'predict':
        if not args.text:
            print("❌ Error: --text argument is required for prediction")
            sys.exit(1)
        
        # Predict sentiment
        result = predict_sentiment(args.text, args.models_dir)
        
        print("\n" + "=" * 80)
        print("🔮 SENTIMENT PREDICTION")
        print("=" * 80)
        print(f"\n📝 Review: {args.text}")
        print(f"\n😊 Sentiment: {result['sentiment']}")
        
        if result['confidence']:
            print(f"📊 Confidence: {result['confidence']:.2%}")
            print("\n📈 Probability Distribution:")
            for sentiment, prob in result['probabilities'].items():
                print(f"   {sentiment}: {prob:.2%}")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
