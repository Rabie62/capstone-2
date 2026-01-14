
from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
import os

app = Flask(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.model = tf.keras.models.load_model('models/best_model.h5')
        with open('models/tokenizer.pkl', 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.labels = {0: 'Negative', 1: 'Somewhat Negative', 2: 'Neutral', 3: 'Somewhat Positive', 4: 'Positive'}
        
    def predict(self, text):
        import re
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        sequence = self.tokenizer.texts_to_sequences([text])
        sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)
        pred = self.model.predict(sequence)[0]
        predicted_class = np.argmax(pred)
        return self.labels[predicted_class], float(pred[predicted_class])

analyzer = SentimentAnalyzer()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    sentiment, confidence = analyzer.predict(data['text'])
    return jsonify({
        'sentiment': sentiment,
        'confidence': confidence
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)