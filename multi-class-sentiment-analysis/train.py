
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return ' '.join(text.split())

def create_model(vocab_size, max_len=100):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 100, input_length=max_len),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(args):
    os.makedirs('models', exist_ok=True)
    
    # Load data
    train_df = pd.read_csv(r"C:/Desktop/ML bootcamp/multi-class-sentiment-analysis/data/train/train.tsv", sep='\t', header=0)
    
    # Preprocess
    tokenizer = Tokenizer(num_words=args.max_features, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_df['Phrase'].apply(clean_text))
    
    X = tokenizer.texts_to_sequences(train_df['Phrase'].apply(clean_text))
    X = pad_sequences(X, maxlen=args.max_len, padding='post', truncating='post')
    y = train_df['Sentiment'].values
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    model = create_model(len(tokenizer.word_index) + 1)
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=args.epochs, batch_size=args.batch_size,
              callbacks=[EarlyStopping(patience=5)], verbose=1)
    
    # Save
    model.save('models/best_model.h5')
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print("Training completed!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_features', type=int, default=20000)
    parser.add_argument('--max_len', type=int, default=100)
    args = parser.parse_args()
    
    train_model(args)

if __name__ == "__main__":
    main()
