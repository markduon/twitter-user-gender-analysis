#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
from nltk.corpus import words
nltk.download('words')
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings
warnings.simplefilter("ignore") # Ignore all warnings
from sklearn.metrics import confusion_matrix
import string
import re
from wordcloud import WordCloud
import pickle # to save the vectorizer and the trained model
from collections import Counter
from sklearn.metrics import confusion_matrix


# ## PROCEED THE TRAIN SET 

# In[2]:


# now going back to the non-golden data (only 20k row), 
user_df = pd.read_csv('data/train_data.csv', encoding='latin-1')
user_df.describe()


# In[6]:


# Get the unique values in the 'is_bot' column
unique_label = user_df['is_bot'].unique()

# Print the unique values
print(unique_label)


# In[7]:


# plot the human and non-human counts
ax = user_df['is_bot'].value_counts().plot(kind='bar', figsize=(1.5,3.5))
fig = ax.get_figure()
ax.set_title("Twitter User Dataset")
ax.set_xlabel('is_bot')
ax.set_ylabel('Value Counts');


# ### Remove columns

# In[3]:


user_df["is_bot"].value_counts()


# In[4]:


column_names = list(user_df.columns)

# Print the list of column names
print(column_names)


# ### Drop the unnecessary columns as we only need text column

# In[5]:


# now drop the unneccesary columns
drop_columns = ['fav_number','name','profileimage','retweet_count',
            'tweet_count','date_created','date_last_judged','days_active',
            'tweet_rate','retweet_rate', 'fav_rate']

# drop others columns as we only do text analysis
proceed_user_df = user_df.drop(columns=drop_columns)


# In[9]:


column= list(proceed_user_df.columns)

# Print the list of column names
print(column)


# # BAG OF WORDS

# ## Preprocess before applying bag of words

# In[8]:


# Preprocess the tweet data 
punctuation = string.punctuation

# a function to clean tweets
def cleanTweet(tweet):
    # Eliminate HTML special entities (e.g. &amp;)
    tweet = re.sub(r'\&\w*;', '', tweet)
    # Change @username to AT_USER
    tweet = re.sub('@[^\s]+','',tweet)
    # Handle tickers by removing the $
    tweet = re.sub(r'\$\w*', '', tweet)
    # To lowercase by converting the tweets to lowercase
    tweet = tweet.lower()
    # Handle hyperlinks by removing the link entirely
    tweet = re.sub(r'https?:\/\/.*\/\w*', '', tweet)
    # Deal with hashtags by removing '#' sign and the word following it
    tweet = re.sub(r'#\w*', '', tweet)
    # Eliminate Punctuation and split 's, 't, 've with a space for filter
    tweet = re.sub(r'[' + punctuation.replace('@', '') + ']+', ' ', tweet)
    # Eliminate numbers
    tweet = re.sub(r'\d', '', tweet)
    # Eliminate words with 2 or fewer letters
    tweet = re.sub(r'\b\w{1,2}\b', '', tweet)
    # Eliminate whitespace (including new line characters)
    tweet = re.sub(r'\s\s+', ' ', tweet)
    # Eliminate characters beyond Basic Multilingual Plane (BMP) of Unicode:
    tweet = ''.join(c for c in tweet if c <= '\uFFFF') 
    # Remove non english character
    tweet = re.sub('[^\x00-\x7F]', '',tweet)
    # Remove repeated space
    tweet = re.sub(r'\s+', ' ', tweet)
    # Remove repeated characters
    tweet = re.sub(r'(.)\1+', r'\1\1',tweet)
    # Trim space at the beginning and end of tweet
    tweet = tweet.strip()
    # Remove stopwords
    tweet = ' '.join(word for word in tweet.split() if word not in stopwords.words('english'))
    

    return tweet
# ______________________________________________________________
# clean dataframe's tweet column
proceed_user_df['text'] = proceed_user_df['text'].apply(cleanTweet)

# Replace null values in "description" column with empty strings
proceed_user_df['description'].fillna('', inplace=True)

# clean dataframe's tweet column
proceed_user_df['description'] = proceed_user_df['description'].apply(cleanTweet)


# ### Combine text and description column

# In[9]:


# Combine "text" and "description" columns
proceed_user_df['combined_text'] = proceed_user_df['description'] + ' ' + proceed_user_df['text']


# In[10]:


# changing df to list of string preparing for the corpus
corpus = proceed_user_df['combined_text'].to_list()
label = proceed_user_df['is_bot'].to_list()


# ### Remove  records in the corpus with empty words, making into new corpus and new label
# 

# In[11]:


#remove records in the corpus with empty words
def contains_word(s):
    return any(i.isalpha() for i in s.split())

# Make a new corpus and label
new_corpus = []
new_label = []
# remove empty record
for i, v in enumerate(corpus):
    if contains_word(v):
        new_corpus.append(v) # adding value of the corpus to the new_corpus
        new_label.append(label[i]) # adding index of the label to the new_label


# In[1]:


new_corpus


# In[15]:


# check the length of both corpus and label
len(new_corpus), len(new_label)


# ## Apply countVectorizer()

# In[13]:


vectorizer = CountVectorizer() #ngram_range=(1, 2)
X = vectorizer.fit_transform(new_corpus)


# In[14]:


# wordcloud on the features names
feature_names = vectorizer.get_feature_names_out()

# Generate a word cloud image
wordcloud = WordCloud(width=1600, height=800, background_color='white',random_state=22).generate(' '.join(feature_names))

# Display the generated word cloud image
plt.figure(figsize=(20, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# In[15]:


X.shape 


# In[16]:


# Split data into two sets based on labels (0 and 1)
human_text = [new_corpus[i] for i in range(len(new_label)) if new_label[i] == 0]
bot_text = [new_corpus[i] for i in range(len(new_label)) if new_label[i] == 1]

human_text_combined = ' '.join(human_text)
bot_text_combined = ' '.join(bot_text)

# Create word cloud for human accounts
wordcloud_human = WordCloud(width=1600, height=800, background_color='white',random_state=22).generate(human_text_combined)

# Create word cloud for bot accounts
wordcloud_bot = WordCloud(width=1600, height=800, background_color='white',random_state=22).generate(bot_text_combined)

# Plot word cloud for human accounts
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_human, interpolation='bilinear')
plt.title("Word Cloud for Human Accounts")
plt.axis('off')
plt.show()

# Plot word cloud for bot accounts
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_bot, interpolation='bilinear')
plt.title("Word Cloud for Bot Accounts")
plt.axis('off')
plt.show()


# In[17]:


# Function to get the top N words from a text
def get_top_words(text, n=10):
    word_count = Counter(text.split())
    top_words = word_count.most_common(n)
    return top_words

# Get the top 10 words for human and bot categories
before_top_words_human = get_top_words(human_text_combined, n=10)
before_top_words_bot = get_top_words(bot_text_combined, n=10)

# # Plot the top 10 words for each category
plt.figure(figsize=(16, 8))

# Sort the words and counts in descending order
sorted_words_human = [word for word, count in before_top_words_human][::-1]
sorted_counts_human = [count for word, count in before_top_words_human][::-1]

sorted_words_bot = [word for word, count in before_top_words_bot][::-1]
sorted_counts_bot = [count for word, count in before_top_words_bot][::-1]

# Plot top words for humans
plt.subplot(1, 2, 1)
plt.barh(sorted_words_human, sorted_counts_human)
plt.xlabel("Word Count")
plt.title("Top 10 Words for Human")

# Plot top words for bots
plt.subplot(1, 2, 2)
plt.barh(sorted_words_bot, sorted_counts_bot)
plt.xlabel("Word Count")
plt.title("Top 10 Words for Bot")

plt.tight_layout()
plt.show()


# ## Train the LogisticRegression model

# In[18]:


# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, new_label, test_size=0.2, random_state=1)

# Train a classifier 
clf = LogisticRegression(penalty='l1', solver='liblinear', C=1.0,random_state=1) # finetune model
clf.fit(X_train, y_train)

# Make predictions on the validation set
y_pred = clf.predict(X_val)

report = classification_report(y_val, y_pred)
print(report)


# In[19]:


# Calculate the confusion matrix
conf_matrix_train = confusion_matrix(y_val, y_pred)

# Extract TP, FP, TN, FN
TP = conf_matrix_train[1, 1]  # True Positives
FP = conf_matrix_train[0, 1]  # False Positives
TN = conf_matrix_train[0, 0]  # True Negatives
FN = conf_matrix_train[1, 0]  # False Negatives

# Print the values
print(f'True Positives (TP): {TP}')
print(f'False Positives (FP): {FP}')
print(f'True Negatives (TN): {TN}')
print(f'False Negatives (FN): {FN}')


# ## TEST WITH THE TEST DATA

# In[20]:


# load the test data from test_data.csv
test_df = pd.read_csv('data/test_data.csv', encoding='latin-1')


# In[21]:


# Pre-process the tweet of the test data
test_df['text'] = test_df['text'].apply(cleanTweet)
# preview some cleaned tweets
test_df['text'].head()


# In[22]:


# Replace null values in "description" column with empty strings
test_df['description'].fillna('', inplace=True)
# clean dataframe's description column
test_df['description'] = test_df['description'].apply(cleanTweet)
# preview some cleaned tweets
test_df['description'].head()


# In[23]:


# Combine "text" and "description" columns
test_df['combined_text'] = test_df['description'] + ' ' + test_df['text']


# In[24]:


# Vectorize the text of the test data using the same vectorizer as the training data
X_test = vectorizer.transform(test_df['combined_text'].tolist())


# In[25]:


X_test.shape


# In[26]:


y_test = test_df['is_bot'].tolist()


# In[27]:


len(y_test)


# In[28]:


# Make predictions on the validation set
y_test_pred = clf.predict(X_test)

# Print the classification report
classification_report_test_set = classification_report(y_test, y_test_pred)

# Print the classification report
print("Classification Report for test set:")
print(classification_report_test_set)


# In[29]:


# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)

# Extract TP, FP, TN, FN
TP = conf_matrix[1, 1]  # True Positives
FP = conf_matrix[0, 1]  # False Positives
TN = conf_matrix[0, 0]  # True Negatives
FN = conf_matrix[1, 0]  # False Negatives

# Print the values
print(f'True Positives (TP): {TP}')
print(f'False Positives (FP): {FP}')
print(f'True Negatives (TN): {TN}')
print(f'False Negatives (FN): {FN}')


# In[30]:


plt.rcParams['font.family'] = 'Times New Roman'

plt.figure(figsize=(6, 4),dpi=300)
plt.imshow(conf_matrix, cmap=plt.cm.Blues, interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()

classes = ['Human', 'Bot']  
tick_marks = [0, 1]
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

for i in range(2):
    for j in range(2):
        color = 'white' if i == 0 and j == 0 else 'black'  # White text for TN, black text for others
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color=color)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


# ## Save the pre-trained model and the pre-trained vectorizer

# In[ ]:


# pickle.dump(clf, open('LR_model.pkl', 'wb'))


# In[ ]:


# pickle.dump(vectorizer, open('bow.pkl', 'wb'))

