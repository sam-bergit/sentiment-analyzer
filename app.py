from flask import Flask, request, render_template
from textblob import TextBlob
import pandas as pd
import os
import datetime
from werkzeug.utils import secure_filename
from rake_nltk import Rake
import nltk

# Ensure that the NLTK stopwords and punkt tokenizer are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Initialize the Flask application
app = Flask(__name__)

# Configure the upload folder for storing uploaded files
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the log_analysis function
def log_analysis(text, sentiment, subjectivity, keywords, user='default_user'):
    # Create the log file if it doesn't exist
    if not os.path.exists('log.csv'):
        log_df = pd.DataFrame(columns=['ID', 'Date/Time', 'Text', 'Sentiment', 'Subjectivity', 'Keywords', 'User', 'Total Uses'])
    else:
        log_df = pd.read_csv('log.csv')

    # Initialize total_uses to 0
    total_uses = 0

    # Determine the new ID for the entry
    if not log_df.empty:
        id = log_df['ID'].max() + 1
        # Count the total uses by the user
        total_uses = log_df[log_df['User'] == user].shape[0] + 1
    else:
        id = 1

    # Get the current date and time
    date_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Create a new log entry
    new_entry = pd.DataFrame([[id, date_time, text, sentiment, subjectivity, keywords, user, total_uses]],
                             columns=['ID', 'Date/Time', 'Text', 'Sentiment', 'Subjectivity', 'Keywords', 'User', 'Total Uses'])

    # Append the new entry to the log DataFrame
    log_df = log_df.append(new_entry, ignore_index=True)

    # Save the updated log DataFrame to the CSV file
    log_df.to_csv('log.csv', index=False)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Render the index.html template

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
        results.append((text, sentiment, subjectivity, keywords))  # Store the result as a list of tuples
        log_analysis(text, sentiment, subjectivity, keywords, user='default_user')  # Log the analysis

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

        # Iterate over each text entry in the DataFrame and analyze sentiment, subjectivity, and keywords
        for text in data['text']:
            blob = TextBlob(text)  # Create a TextBlob object for each text
            # Determine the sentiment and subjectivity of the text
            sentiment = 'Positive' if blob.sentiment.polarity > 0 else 'Negative' if blob.sentiment.polarity < 0 else 'Neutral'
            subjectivity = 'Subjective' if blob.sentiment.subjectivity > 0.5 else 'Objective'
            # Extract keywords
            rake.extract_keywords_from_text(text)
            keywords = ', '.join(rake.get_ranked_phrases())
            results.append((text, sentiment, subjectivity, keywords))  # Store the result as a list of tuples
            log_analysis(text, sentiment, subjectivity, keywords, user='default_user')  # Log the analysis
    else:
        return "No input provided"  # If no input is provided, return a message

    # Render the results.html template with the results
    return render_template('results.html', results=results)

# Define the route for displaying the log
@app.route('/log')
def display_log():
    if not os.path.exists('log.csv'):
        return "No log entries found."

    log_df = pd.read_csv('log.csv')
    return render_template('log.html', log_entries=log_df.to_dict(orient='records'))

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
