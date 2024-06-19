from flask import Flask, request, render_template
from textblob import TextBlob
import pandas as pd
import os
from werkzeug.utils import secure_filename
from rake_nltk import Rake
import nltk
from langdetect import detect, LangDetectException
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Ensure that the NLTK stopwords and punkt tokenizer are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the Flask application
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sentiment_log.db'
db = SQLAlchemy(app)

# Configure the upload folder for storing uploaded files
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class SentimentLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date_time = db.Column(db.DateTime, default=datetime.utcnow)
    text = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(50), nullable=False)
    subjectivity = db.Column(db.String(50), nullable=False)
    keywords = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(50), nullable=False)
    user = db.Column(db.String(50), nullable=False)
    total_uses = db.Column(db.Integer, nullable=False, default=1)

    def __repr__(self):
        return f"SentimentLog('{self.date_time}', '{self.sentiment}', '{self.subjectivity}', '{self.keywords}', '{self.language}', '{self.user}', '{self.total_uses}')"

# Ensure the database is created within the application context
with app.app_context():
    db.create_all()

# Define a function to detect the language of a text
def detect_language(text):
    try:
        if len(text) < 10:  # Set a minimum text length for detection
            return "Unknown"
        return detect(text)
    except LangDetectException:
        return "Unknown"

# Define a function to log the analysis
def log_analysis(text, sentiment, subjectivity, keywords, language, user='default_user'):
    existing_entry = SentimentLog.query.filter_by(text=text, user=user).first()
    if existing_entry:
        existing_entry.total_uses += 1
    else:
        new_entry = SentimentLog(
            text=text,
            sentiment=sentiment,
            subjectivity=subjectivity,
            keywords=keywords,
            language=language,
            user=user
        )
        db.session.add(new_entry)
    db.session.commit()

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for analyzing sentiment, subjectivity, and keyword extraction
@app.route('/analyze', methods=['POST'])
def analyze():
    results = []
    rake = Rake()  # Initialize RAKE

    # Check if text is submitted through the form
    if 'text' in request.form and request.form['text'].strip() != "":
        text = request.form['text']  # Get the text from the form
        blob = TextBlob(text)  # Create a TextBlob object
        # Determine the sentiment and subjectivity of the text
        sentiment = 'Positive' if blob.sentiment.polarity > 0 else 'Negative' if blob.sentiment.polarity < 0 else 'Neutral'
        subjectivity = 'Subjective' if blob.sentiment.subjectivity > 0.5 else 'Objective'
        # Extract keywords
        rake.extract_keywords_from_text(text)
        keywords = ', '.join(rake.get_ranked_phrases())
        # Detect language
        language = detect_language(text)
        results.append((text, sentiment, subjectivity, keywords, language))  # Store the result as a list of tuples
        log_analysis(text, sentiment, subjectivity, keywords, language, user='bergersam')

    # Check if a file is submitted through the form
    elif 'file' in request.files:
        file = request.files['file']  # Get the file from the form
        if file.filename == '':
            return "No selected file"  # If no file is selected, return a message
        filename = secure_filename(file.filename)  # Secure the filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)  # Define the file path
        file.save(file_path)  # Save the file to the upload folder
        data = pd.read_csv(file_path)  # Read the CSV file into a DataFrame

        # Check if the required 'text' column is present
        if 'text' not in data.columns:
            return "CSV file must have a 'text' column"  # Return a message if the 'text' column is missing

        # Iterate over each text entry in the DataFrame and analyze sentiment, subjectivity, keywords, and language
        for text in data['text']:
            blob = TextBlob(text)  # Create a TextBlob object for each text
            # Determine the sentiment and subjectivity of the text
            sentiment = 'Positive' if blob.sentiment.polarity > 0 else 'Negative' if blob.sentiment.polarity < 0 else 'Neutral'
            subjectivity = 'Subjective' if blob.sentiment.subjectivity > 0.5 else 'Objective'
            # Extract keywords
            rake.extract_keywords_from_text(text)
            keywords = ', '.join(rake.get_ranked_phrases())
            # Detect language
            language = detect_language(text)
            results.append((text, sentiment, subjectivity, keywords, language))  # Store the result as a list of tuples
            log_analysis(text, sentiment, subjectivity, keywords, language, user='bergersam')
    else:
        return "No input provided"  # If no input is provided, return a message

    # Render the results.html template with the results
    return render_template('results.html', results=results)

# Define the route for displaying the log
@app.route('/log')
def display_log():
    log_entries = SentimentLog.query.all()
    return render_template('log.html', log_entries=log_entries)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
