from nltk.corpus import stopwords
import string
import re
# Preprocess the tweet data 
punctuation = string.punctuation

# a function to clean tweets
def cleanTweet(tweet):
    # Remove HTML special entities (e.g. &amp;)
    tweet = re.sub(r'\&\w*;', '', tweet)
    #Convert @username to AT_USER
    tweet = re.sub('@[^\s]+','',tweet)
    # Remove tickers
    tweet = re.sub(r'\$\w*', '', tweet)
    # To lowercase
    tweet = tweet.lower()
    # Remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#\w*', '', tweet)
    # Remove Punctuation and split 's, 't, 've with a space for filter
    tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)
    # Remove numbers
    tweet = re.sub(r'\d', '', tweet)
    # Remove words with 2 or fewer letters
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    # Remove whitespace (including new line characters)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet = ''.join(c for c in tweet if c <= '\uFFFF') 
    # Remove non english character
    tweet = re.sub('[^\x00-\x7F]', '',tweet)
    # remove repeated space
    tweet = re.sub(r'\s+', ' ', tweet)
    # remove repeated characters
    tweet = re.sub(r'(.)\1+', r'\1\1',tweet)
    # trim space at the beginning and end of tweet
    tweet = tweet.strip()
    # remove stopwords
    tweet = ' '.join(word for word in tweet.split() if word not in stopwords.words('english'))
    

    return tweet
# ______________________________________________________________    