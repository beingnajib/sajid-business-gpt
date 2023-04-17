import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import sqlite3
import requests
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse

# Define the problem statement
problem_statement = "Develop an AI-powered chatbot that can provide financial calculations, data analysis, and news updates for the global textile industry."

# Preprocess the problem statement
lemmatizer = WordNetLemmatizer()
tokenized_words = nltk.word_tokenize(problem_statement.lower())
preprocessed_statement = ' '.join([lemmatizer.lemmatize(word) for word in tokenized_words])

# Define sample user queries and responses
user_queries = ['What is the current inventory level?',
                'Can you provide sales forecast for next quarter?',
                'What is the stock price of our company?',
                'What are the latest global textile industry news updates?']

chatbot_responses = ['The current inventory level is 10,000 units.',
                     'Based on our data analysis, we predict a 15% increase in sales next quarter.',
                     'The stock price of our company is $50 per share.',
                     'Here are the latest news updates related to the global textile industry:']

# Create and connect to a SQLite database
conn = sqlite3.connect('chatbot_data.db')

# Define the SQL commands to create tables
create_inventory_table = "CREATE TABLE IF NOT EXISTS inventory (id INTEGER PRIMARY KEY, product_name TEXT, quantity INTEGER);"
create_sales_table = "CREATE TABLE IF NOT EXISTS sales (id INTEGER PRIMARY KEY, date TEXT, amount REAL);"
create_stock_price_table = "CREATE TABLE IF NOT EXISTS stock_price (id INTEGER PRIMARY KEY, date TEXT, price REAL);"

# Execute the SQL commands
conn.execute(create_inventory_table)
conn.execute(create_sales_table)
conn.execute(create_stock_price_table)

# Load data from the database
inventory_df = pd.read_sql_query("SELECT * FROM inventory", conn)
sales_df = pd.read_sql_query("SELECT * FROM sales", conn)
stock_price_df = pd.read_sql_query("SELECT * FROM stock_price", conn)

# Train the chatbot using Naive Bayes classifier
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(user_queries)
y_train = chatbot_responses
clf = MultinomialNB().fit(X_train, y_train)

# Define functions to handle user queries
def get_inventory_level():
    return f"The current inventory level is {inventory_df['quantity'].sum()} units."

def predict_sales_forecast():
    # perform data analysis on sales data using machine learning models
    return "Based on our data analysis, we predict a 15% increase in sales next quarter."

def get_stock_price():
    return f"The stock price of our company is ${stock_price_df.iloc[-1]['price']} per share."

def get_textile_news_updates():
    # define API endpoints for textile news channels
    api_endpoints = ['https://newsapi.org/v2/top-headlines?country=us&category=business&q=textile&apiKey=YOUR_API_KEY',
                     'https://api.currentsapi.services/v1/search?keywords=textile&page_number=1&apiKey=YOUR_API_KEY',
                                           'https://gnews.io/api/v4/search?q=textile&token=YOUR_API_KEY']

    # request latest news articles from each channel
    news_articles = []
    for endpoint in api_endpoints:
        response = requests.get(endpoint)
        if response.status_code == 200:
            news_articles.extend(response.json()['articles'])

    # extract article titles and URLs
    headlines = [article['title'] + ': ' + article['url'] for article in news_articles]

    # format response string
    response_str = ' '.join([f"{i+1}. {headline}" for i, headline in enumerate(headlines)])

    return response_str

# Define the chatbot response
def get_chatbot_response(user_query):
    preprocessed_query = ' '.join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(user_query.lower())])
    X_test = vectorizer.transform([preprocessed_query])
    predicted_response = clf.predict(X_test)
    if "inventory level" in preprocessed_query:
        return get_inventory_level()
    elif "sales forecast" in preprocessed_query:
        return predict_sales_forecast()
    elif "stock price" in preprocessed_query:
        return get_stock_price()
    elif "textile news updates" in preprocessed_query or "global textile industry news" in preprocessed_query:
        return get_textile_news_updates()
    else:
        return "I'm sorry, I don't understand your query."

# Configure Flask and Twilio APIs to create a chatbot interface
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/sms", methods=['POST'])
def sms_reply():
    msg = request.form.get('Body')
    resp = MessagingResponse()
    reply = get_chatbot_response(msg)
    resp.message(reply)
    return str(resp)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=80)
