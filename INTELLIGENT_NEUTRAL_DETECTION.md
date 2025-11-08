# Intelligent Neutral Detection for Sentiment Analysis

## Overview
Enhanced the SVM model with intelligent neutral sentiment detection to handle mixed sentiment reviews (e.g., "camera quality is good but sound quality is poor").

## Key Improvements

### 1. **SVM with Probability Estimates**
- Changed SVM configuration from `SVC(kernel='linear', random_state=42)` 
- To: `SVC(kernel='linear', probability=True, random_state=42)`
- This enables the model to provide confidence scores for each sentiment class

### 2. **Intelligent Mixed Sentiment Detection**
The enhanced prediction function now analyzes reviews for:

#### a) **Keyword Analysis**
- **Positive keywords**: good, great, excellent, amazing, love, best, perfect, nice, awesome, fantastic, wonderful
- **Negative keywords**: bad, poor, terrible, worst, hate, awful, horrible, disappointing, useless, waste
- **Contrast words**: but, however, though, although, except, yet, while

#### b) **Automatic Neutral Classification**
A review is classified as **Neutral** if:
1. **Contains both positive AND negative keywords** (mixed sentiment)
2. **Has contrast words AND mixed sentiments** (e.g., "good but bad")
3. **Low confidence difference**: If positive and negative probabilities are close (within 30%) and maximum probability is less than 70%

#### c) **Probability Adjustment**
When mixed sentiment is detected:
- Neutral probability is boosted to 50%+ 
- Positive and negative probabilities are redistributed proportionally
- This provides more accurate confidence scores

## Test Results

### ✅ Mixed Sentiment Examples (Now Correctly Classified as Neutral)

1. **"camera quality is good but sound quality is poor"**
   - Sentiment: **Neutral** ✓
   - Confidence: 51.11%
   - Distribution: Negative (47.07%), Neutral (51.11%), Positive (1.82%)

2. **"great battery life but terrible display quality"**
   - Sentiment: **Neutral** ✓
   - Confidence: 50.07%
   - Distribution: Negative (0.12%), Neutral (50.07%), Positive (49.81%)

3. **"nice design and good build quality however performance is disappointing"**
   - Sentiment: **Neutral** ✓
   - Confidence: 51.13%
   - Distribution: Negative (1.84%), Neutral (51.13%), Positive (47.03%)

### ✅ Clear Sentiment Examples (Still Accurate)

4. **"excellent product, highly recommended"**
   - Sentiment: **Positive** ✓
   - Confidence: 95.16%
   - Distribution: Negative (2.73%), Neutral (2.11%), Positive (95.16%)

5. **"worst purchase ever, complete waste of money"**
   - Sentiment: **Negative** ✓
   - Confidence: 99.97%
   - Distribution: Negative (99.97%), Neutral (0.03%), Positive (0.00%)

## How It Works

### Detection Logic Flow
```
1. Clean and vectorize the review text
2. Get initial prediction from SVM model
3. Extract probability distribution (Negative, Neutral, Positive)
4. Analyze text for sentiment indicators:
   - Count positive keywords
   - Count negative keywords
   - Check for contrast words
5. If mixed sentiment detected:
   - Override prediction to Neutral (class 1)
   - Adjust probability distribution
6. Return final sentiment with confidence scores
```

## Implementation Details

### Files Modified
1. **`main.py`** - Updated `predict_sentiment()` function with intelligent detection
2. **`app/app.py`** - Updated `predict_sentiment()` function for Streamlit app

### Algorithm Parameters
- **Confidence threshold**: 0.3 (for probability difference)
- **Maximum probability threshold**: 0.7
- **Neutral boost factor**: 0.5 + (minority_sentiment_proportion * 0.3)

## Benefits

1. **Better Real-World Performance**: Handles nuanced reviews with both pros and cons
2. **Balanced Predictions**: Reduces false positives/negatives for mixed reviews
3. **Transparent Confidence**: Provides probability distribution for each class
4. **Context-Aware**: Uses linguistic patterns (contrast words) to identify mixed sentiments
5. **Maintains Accuracy**: Doesn't affect clear positive/negative sentiment detection

## Model Performance
- **SVM Accuracy**: 94.69% (maintained from original)
- **Precision**: 94.75%
- **Recall**: 94.69%
- **F1-Score**: 94.46%

## Usage

### Command Line
```bash
python main.py predict --text "camera quality is good but sound quality is poor"
```

### Streamlit App
```bash
streamlit run app/app.py
```
Then enter your review text in the web interface.

## Future Enhancements
1. Add more domain-specific keywords for product reviews
2. Implement aspect-based sentiment analysis (separate scores for camera, battery, etc.)
3. Use word embeddings to detect semantic similarities beyond keyword matching
4. Train a separate neutral class classifier for edge cases

---
**Updated**: October 27, 2025
**Authors**: Anshul Jadwani, Harshil Patni, Dhruv Hirani
