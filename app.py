from flask import Flask, request, render_template
from textblob import TextBlob
import pandas as pd
import os
from werkzeug.utils import secure_filename
from rake_nltk import Rake
import nltk
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Ensure that the NLTK stopwords and punkt tokenizer are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the Flask application
app = Flask(__name__)

# Configure the upload folder for storing uploaded files
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure the SQLAlchemy part of the app instance
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///historical_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Create the SQLAlchemy db instance
db = SQLAlchemy(app)

# Define the HistoricalData model
class HistoricalData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_time = db.Column(db.DateTime, default=datetime.utcnow)
    text = db.Column(db.String(500))
    sentiment = db.Column(db.String(20))
    subjectivity = db.Column(db.String(20))
    keywords = db.Column(db.String(500))
    user = db.Column(db.String(50), default='default_user')

# Create the database tables
with app.app_context():
    db.create_all()

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for analyzing sentiment, subjectivity, and keyword extraction
@app.route('/analyze', methods=['POST'])
def analyze():
    results = []
    rake = Rake()

    if 'text' in request.form and request.form['text'].strip() != "":
        text = request.form['text']
        blob = TextBlob(text)
        sentiment = 'Positive' if blob.sentiment.polarity > 0 else 'Negative' if blob.sentiment.polarity < 0 else 'Neutral'
        subjectivity = 'Subjective' if blob.sentiment.subjectivity > 0.5 else 'Objective'
        rake.extract_keywords_from_text(text)
        keywords = ', '.join(rake.get_ranked_phrases())
        results.append((text, sentiment, subjectivity, keywords))
        log_analysis(text, sentiment, subjectivity, keywords)

    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        data = pd.read_csv(file_path)

        if 'text' not in data.columns:
            return "CSV file must have a 'text' column"

        for text in data['text']:
            blob = TextBlob(text)
            sentiment = 'Positive' if blob.sentiment.polarity > 0 else 'Negative' if blob.sentiment.polarity < 0 else 'Neutral'
            subjectivity = 'Subjective' if blob.sentiment.subjectivity > 0.5 else 'Objective'
            rake.extract_keywords_from_text(text)
            keywords = ', '.join(rake.get_ranked_phrases())
            results.append((text, sentiment, subjectivity, keywords))
            log_analysis(text, sentiment, subjectivity, keywords)
    else:
        return "No input provided"

    return render_template('results.html', results=results)

def log_analysis(text, sentiment, subjectivity, keywords, user='default_user'):
    new_entry = HistoricalData(
        text=text,
        sentiment=sentiment,
        subjectivity=subjectivity,
        keywords=keywords,
        user=user
    )
    db.session.add(new_entry)
    db.session.commit()

@app.route('/log')
def display_log():
    log_entries = HistoricalData.query.all()
    return render_template('log.html', log_entries=log_entries)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
