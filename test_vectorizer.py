import joblib
from pathlib import Path

# Load vectorizer
v = joblib.load(Path('models/tfidf_vectorizer.pkl'))
print(f'Vectorizer type: {type(v)}')
print(f'Has idf_: {hasattr(v, "idf_")}')
print(f'Has vocabulary_: {hasattr(v, "vocabulary_")}')

if hasattr(v, 'vocabulary_'):
    print(f'Vocabulary size: {len(v.vocabulary_)}')

# Try to transform a test text
try:
    from utils.preprocessing import clean_text
    test_text = "This product is good"
    cleaned = clean_text(test_text)
    print(f'\nTest text: {test_text}')
    print(f'Cleaned text: {cleaned}')
    
    result = v.transform([cleaned])
    print(f'\nTransform successful!')
    print(f'Result shape: {result.shape}')
except Exception as e:
    print(f'\nError during transform: {e}')
