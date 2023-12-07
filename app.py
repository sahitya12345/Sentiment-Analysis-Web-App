from flask import Flask, render_template, request, jsonify
import pickle
from keras.utils import pad_sequences
from keras.models import load_model
from keras.datasets import imdb
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
import nltk
import numpy as np


app = Flask(__name__)

# Load the logistic regression model from the pickle file
with open('logistic_regression_model.pkl', 'rb') as file:
    logistic_regression = pickle.load(file)

# Load the Keras model
model = load_model('sentiment_model.h5')

stopwords = set(nltk.corpus.stopwords.words('english'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        user_review = request.form['review']

        # Tokenize and pad the input sequence
        sequence = imdb.get_word_index()
        user_review_seq = [sequence[word] if word in sequence and sequence[word] < 10000 else 0 for word in user_review.split()]
        user_review_padded = pad_sequences([user_review_seq], maxlen=200)

        # Get Bi-GRU features
        bi_gru_features = model.predict(user_review_padded)

        # Flatten the features to 2D
        num_samples, sequence_length, num_features = bi_gru_features.shape
        bi_gru_features_2d = bi_gru_features.reshape(num_samples, sequence_length * num_features)

        collocations = extract_collocations(user_review)

        prediction = logistic_regression.predict(bi_gru_features_2d)[0]

        sentiment_intensity = np.mean(bi_gru_features)

        sentiment_score = calculate_sentiment_score(sentiment_intensity)

        metrics = {
            'prediction': int(prediction),
            'sentiment_intensity': float(sentiment_intensity),
            'sentiment_score': float(sentiment_score),
            'word_count': len(user_review.split()),
            'collocations': collocations,
        }

        return jsonify(metrics)

    except Exception as e:
        return jsonify({'error': str(e)})

def calculate_sentiment_score(sentiment_intensity):

    if sentiment_intensity >= 0.7:
        return 1.0  # Highly Positive
    elif 0.5 <= sentiment_intensity < 0.7:
        return 0.5  # Moderately Positive
    elif 0.3 <= sentiment_intensity < 0.5:
        return 0.0  # Neutral
    elif 0.1 <= sentiment_intensity < 0.3:
        return -0.5  # Moderately Negative
    else:
        return -1.0  # Highly Negative

def extract_collocations(sentence):

    words = [word.lower() for word in nltk.word_tokenize(sentence) if word.isalnum() and word.lower() not in stopwords]
    
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(words)
    collocations = finder.nbest(bigram_measures.pmi, 5)  
    
    return collocations
    
if __name__ == '__main__':
    app.run(debug=True)