from flask import Flask, request, render_template
from textblob import TextBlob
import pandas as pd
import os
from werkzeug.utils import secure_filename
from rake_nltk import Rake
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Add language detection
from langdetect import detect

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiment_analysis.db'
db = SQLAlchemy(app)

class AnalysisLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_time = db.Column(db.String(50))
    text = db.Column(db.Text)
    sentiment = db.Column(db.String(50))
    subjectivity = db.Column(db.String(50))
    keywords = db.Column(db.Text)
    language = db.Column(db.String(50))
    character_count = db.Column(db.Integer)
    user = db.Column(db.String(50))
    total_uses = db.Column(db.Integer)

with app.app_context():
    db.create_all()

def log_analysis(text, sentiment, subjectivity, keywords, language, character_count, user='default_user'):
    date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    total_uses = AnalysisLog.query.filter_by(user=user).count() + 1
    new_entry = AnalysisLog(
        date_time=date_time,
        text=text,
        sentiment=sentiment,
        subjectivity=subjectivity,
        keywords=keywords,
        language=language,
        character_count=character_count,
        user=user,
        total_uses=total_uses
    )
    db.session.add(new_entry)
    db.session.commit()

@app.route('/')
def home():
    return render_template('index.html')

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
        language = detect(text)
        character_count = len(text)
        results.append((text, sentiment, subjectivity, keywords, language, character_count))
        log_analysis(text, sentiment, subjectivity, keywords, language, character_count, user='bergersam')

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
            language = detect(text)
            character_count = len(text)
            results.append((text, sentiment, subjectivity, keywords, language, character_count))
            log_analysis(text, sentiment, subjectivity, keywords, language, character_count, user='bergersam')
    else:
        return "No input provided"

    return render_template('results.html', results=results)

@app.route('/log')
def display_log():
    log_entries = AnalysisLog.query.all()
    return render_template('log.html', log_entries=log_entries)

if __name__ == '__main__':
    app.run(debug=True)
