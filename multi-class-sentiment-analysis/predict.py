
import argparse
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt', quiet=True)

def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return ' '.join(text.split())

class SentimentPredictor:
    def __init__(self, model_path, tokenizer_path):
        self.model = tf.keras.models.load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.labels = {0: 'Negative', 1: 'Somewhat Negative', 2: 'Neutral', 3: 'Somewhat Positive', 4: 'Positive'}
        
    def predict(self, text):
        cleaned = clean_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        sequence = pad_sequences(sequence, maxlen=100, padding='post', truncating='post')
        pred = self.model.predict(sequence)[0]
        predicted_class = np.argmax(pred)
        return self.labels[predicted_class], float(pred[predicted_class])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--model_path', type=str, default='models/best_model.h5')
    parser.add_argument('--tokenizer_path', type=str, default='models/tokenizer.pkl')
    args = parser.parse_args()
    
    predictor = SentimentPredictor(args.model_path, args.tokenizer_path)
    sentiment, confidence = predictor.predict(args.text)
    
    print(f"Text: {args.text}")
    print(f"Sentiment: {sentiment}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()