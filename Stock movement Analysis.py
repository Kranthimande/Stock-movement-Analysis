#Code to Scrape Twitter Data

import tweepy
import pandas as pd

# Set up the Twitter API client
consumer_key = 'your_consumer_key'
consumer_secret = 'your_consumer_secret'
access_token = 'your_access_token'
access_token_secret = 'your_access_token_secret'

auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Function to scrape tweets based on a hashtag
def scrape_tweets(query, max_tweets=1000):
    tweets = tweepy.Cursor(api.search, q=query, lang="en", tweet_mode='extended').items(max_tweets)
    tweet_data = []

    for tweet in tweets:
        tweet_data.append({
            'date': tweet.created_at,
            'user': tweet.user.screen_name,
            'text': tweet.full_text,
            'retweets': tweet.retweet_count,
            'favorites': tweet.favorite_count
        })

    return pd.DataFrame(tweet_data)

# Example usage: Scrape tweets related to "stock market"
tweets_df = scrape_tweets('stock market', max_tweets=500)
tweets_df.to_csv('stock_market_tweets.csv', index=False)

#Data Preprocessing

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess tweets
def preprocess_tweet(tweet):
    tweet = re.sub(r'http\S+', '', tweet)  # Remove URLs
    tweet = re.sub(r'@\S+', '', tweet)  # Remove mentions
    tweet = re.sub(r'#\S+', '', tweet)  # Remove hashtags
    tweet = re.sub(r'[^A-Za-z0-9\s]', '', tweet)  # Remove special characters
    tweet = tweet.lower()  # Convert to lowercase
    words = word_tokenize(tweet)
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Apply preprocessing
tweets_df['cleaned_text'] = tweets_df['text'].apply(preprocess_tweet)


#Sentiment Analysis code
from textblob import TextBlob

# Function to get sentiment polarity
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Returns a value between -1 and 1
    return sentiment

# Apply sentiment analysis
tweets_df['sentiment'] = tweets_df['cleaned_text'].apply(get_sentiment)

# For visualization, categorize sentiments into positive, negative, and neutral
def sentiment_category(score):
    if score > 0:
        return 'positive'
    elif score < 0:
        return 'negative'
    else:
        return 'neutral'

tweets_df['sentiment_category'] = tweets_df['sentiment'].apply(sentiment_category)

# Display results
tweets_df[['text', 'sentiment', 'sentiment_category']].head()

#Prediction Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Assume you have historical stock data (can be manually labeled or fetched from a stock API)
# Example stock movements (1 = Up, 0 = Down)
# In practice, you would align these with the timestamps of the tweets.

# Let's create a dummy target variable for stock movement prediction (this is just for the example)
tweets_df['stock_movement'] = [1 if i % 2 == 0 else 0 for i in range(len(tweets_df))]  # Dummy values

# Features and target variable
X = tweets_df[['sentiment']]  # We can also use additional features
y = tweets_df['stock_movement']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

#Evaluation and Improvement
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
