import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import string
import re
import scipy
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

import warnings
warnings.filterwarnings("ignore")

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

"""We read the csv to makes a dataframe to work on"""
def preprocess_df(target):
    df = pd.read_csv("processed_twitter_data_with_index.csv", encoding = "ISO-8859-1")
    target_df = target
    
    df = df.set_index("Unnamed: 0", verify_integrity=True)
    target_df = target_df.set_index("Unnamed: 0", verify_integrity=True)

    """description column has some missing columns, fill them with empyt space strings ('')."""

    df['description'] = df['description'].apply(lambda x: '' if pd.isnull(x) else x)

    basic_latin = [chr(i) for i in range(32, 126+1)]
    def is_basic_latin_only(string):
      basic_latin_only = 1
      for i in range(len(string)):
        if not (string[i] in basic_latin):
          basic_latin_only = 0
      return basic_latin_only
      
    def filter_basic_latin(string):
      filter_char_list = []
      for i in range(len(string)):
        if not (string[i] in basic_latin):
          filter_char_list.append(string[i])
      for char in filter_char_list:
        string = re.sub(r"\S*"+char+r"\S*", "", string)
      return string

    def token_column(func_df, column, token_func):
      tokens_list = []
      for i in func_df.index:
        column_i = func_df.loc[i, column]
        #convert all to lower case/Capital Case/UPPER case(?) for easier comparison later
        column_i = column_i.lower()
        column_tokens_i = token_func(column_i)
        tokens_list.append(column_tokens_i)
      return tokens_list


    from nltk.tokenize import TweetTokenizer

    """Change words based off lemmatisation."""
    def lemmatization_filter_column(func_df, token_column):
      filtered_token_list_list = []
      for i in func_df.index:
        filtered_token_list = []
        token_list_i = func_df.loc[i, token_column]
        for word in token_list_i:
          pos_list = [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]
          lemma_word = word
          j = 0
          while (lemma_word == word) and j<len(pos_list):
            if (word in ['was', 'has']):
              break
            lemma_word = WordNetLemmatizer().lemmatize(word, pos=pos_list[j])
            j += 1
          filtered_token_list.append(lemma_word)
        filtered_token_list_list.append(filtered_token_list)
      return filtered_token_list_list

    """list of stopwords to consider filtering"""

    my_stopwords = stopwords.words('english')
    #add anymore you think are legitimate and missing
    my_stopwords.extend(["i'd", "i'm", "i've", "n't"])
    #remove really common words/topics not already in stopwords
    generic_words = ["get", "make", "like", "look",
                         "come", "love", "weather",
                         "see", "follow", "go",
                         "one", "new", "best", "good"]
    my_stopwords.extend(generic_words)
    my_stopwords.extend(["time", "year", "month", "week", "day"]) #time periods
      #due to bad grammar of Tweets, add in contractions that leave out the apostrophe
      #in the destructive tokenization, the end of a contraction is prepended with an apostrophe -- reflect this
    L = len(my_stopwords)
    for i in range(L):
      if "'" in my_stopwords[i]:
        temp_word = my_stopwords[i].replace("'", "")
        my_stopwords.append(temp_word)

        temp_word = my_stopwords[i][my_stopwords[i].index("'"):]
        my_stopwords.append(temp_word)
    my_stopwords = list(set(my_stopwords))

    """Create a filtered version via the stopwords"""

    def stopwords_filter_column(func_df, token_column):
      filtered_token_list_list = []
      for i in func_df.index:
        filtered_token_list = []
        token_list_i = func_df.loc[i, token_column]
        for word in token_list_i:
          if not (word in my_stopwords):
            filtered_token_list.append(word)
        filtered_token_list_list.append(filtered_token_list)
      return filtered_token_list_list

    text_df = df[['description', 'name', 'text', 'is_bot', '_golden']]

    text_df["text_has_hyperlink"] = text_df["text"].apply(lambda x: ('http' in x))
    text_df["text"]= text_df["text"].apply(lambda x: re.sub(r"\S*http\S+", "", x))

    text_df["desc_has_hyperlink"] = text_df["description"].apply(lambda x: ('http' in x))
    text_df["description"]= text_df["description"].apply(lambda x: re.sub(r"\S*http\S+", "", x))

    text_df["text_basic_latin_only"] = text_df["text"].apply(lambda x: is_basic_latin_only(x))
    text_df["text"] = text_df["text"].apply(lambda x: filter_basic_latin(x))

    text_df["desc_basic_latin_only"] = text_df["description"].apply(lambda x: is_basic_latin_only(x))
    text_df["description"] = text_df["description"].apply(lambda x: filter_basic_latin(x))

    #filter out number words

    def has_number(string):
      without_number = True
      for i in range(len(string)):
        if (string[i] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
          without_number = False
      if without_number:
        return 0
      else:
        return 1


    text_df["text_has_number"] = text_df["text"].apply(lambda x: has_number(x))

    def filter_number(string):
      filter_char_list = []
      for i in range(len(string)):
        if (string[i] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
          filter_char_list.append(string[i])
      for char in filter_char_list:
        string = re.sub(r"\S*"+char+r"\S*", "", string)
      return string

    text_df["text"] = text_df["text"].apply(lambda x: filter_number(x))

    text_df["desc_has_number"] = text_df["description"].apply(lambda x: has_number(x))
    text_df["description"] = text_df["description"].apply(lambda x: filter_number(x))

    text_df['text_token_casual'] = token_column(text_df, 'text', TweetTokenizer().tokenize)
    text_df['description_token_casual'] = token_column(text_df, 'description', TweetTokenizer().tokenize)

    #token filters

    text_df['text_token_casual_filtered'] = lemmatization_filter_column(text_df, 'text_token_casual')
    text_df['description_token_casual_filtered'] = lemmatization_filter_column(text_df, 'description_token_casual')

    text_df['text_token_casual_filtered'] = stopwords_filter_column(text_df, 'text_token_casual_filtered')
    text_df['description_token_casual_filtered'] = stopwords_filter_column(text_df, 'description_token_casual_filtered')

    text_df['text_casual_filtered'] = text_df['text_token_casual_filtered'].apply(lambda x: ' '.join(x))
    text_df['description_casual_filtered'] = text_df['description_token_casual_filtered'].apply(lambda x: ' '.join(x))

    def combine_tfidf_df(func_df, column1, column2):
      vectorizer1 = TfidfVectorizer()
      vectors1 = vectorizer1.fit_transform(func_df[column1])
      feature_names1 = vectorizer1.get_feature_names_out()
      #dense = vectors.todense()
      #dense_list = dense.tolist()
      """this method wasnt scaling well enough,
      the sparse matrix (crs in scipy) method toarray worked very well"""

      vectorizer2 = TfidfVectorizer()
      vectors2 = vectorizer2.fit_transform(func_df[column2])
      feature_names2 = vectorizer2.get_feature_names_out()
      #dense = vectors.todense()
      #dense_list = dense.tolist()
      """this method wasnt scaling well enough,
      the sparse matrix (crs in scipy) method toarray worked very well"""

      feature_names = np.append(feature_names1, feature_names2)
      vectors = scipy.sparse.hstack((vectors1, vectors2))
      dense_list = vectors.toarray()
      dense_list = dense_list.astype(np.float32)
      df = pd.DataFrame(dense_list, columns=feature_names)
      return df

    casual_tfidf_df = combine_tfidf_df(text_df, 'text_casual_filtered', 'description_casual_filtered')
    casual_tfidf_df.index = text_df.index

    text_df = text_df.drop(['text_casual_filtered', 'description_casual_filtered',
                            'text_token_casual', 'description_token_casual',
                            'description', 'text', 'name'], axis=1)

    """Remove the non-numerical columns to prepare for classification."""
    """As well remove _golden because we're not using it directly"""

    class_text_df = text_df.drop(['_golden', 'text_token_casual_filtered', 'description_token_casual_filtered'],
                                 axis=1)
    class_text_df = casual_tfidf_df.join(class_text_df)
    class_text_df.index = text_df.index
    
    processed_df = class_text_df

    """reduce processed_df down to rows of just target_df"""

    processed_df = processed_df.reindex(target_df.index)
    
    return processed_df

"""Use the labels and the columns transformed into numerical data (mostly 0 and 1s but ah well) to train a classifier.
"""

"""
#######don't mind this, Im just running tests that it works here

train_df = pd.read_csv("train_data.csv")
y_train = train_df['is_bot']
train_df = preprocess_df(train_df)

test_df = pd.read_csv("test_data.csv")
y_test = test_df['is_bot']
test_df = preprocess_df(test_df)

X_train = train_df.drop(['is_bot'], axis = 1) 

X_test = test_df.drop(['is_bot'], axis = 1) 

clf = pickle.load(open('tfidf_model.sav', 'rb'))

y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)

accuracy1 = accuracy_score(y_test, y_test_pred)
print('Accuracy on test set is: %.4f\n' %accuracy1)
matrix = confusion_matrix(y_test, y_test_pred)
print("     Human Bot (Predicted)")
print("Human " + str(matrix[0][0])+ "    " + str(matrix[0][1]))
print("Bot   " + str(matrix[1][0])+ "    " + str(matrix[1][1]))
print("(Actual)")

accuracy2 = accuracy_score(y_train, y_train_pred)
print('Accuracy on train set is: %.4f\n' %accuracy2)
matrix = confusion_matrix(y_train, y_train_pred)
print("     Human Bot (Predicted)")
print("Human " + str(matrix[0][0])+ "    " + str(matrix[0][1]))
print("Bot   " + str(matrix[1][0])+ "    " + str(matrix[1][1]))
print("(Actual)")
"""

def create_clf_model(train, model_name):
    train_df = pd.read_csv(train)
    train_df = preprocess_df(train_df)
    
    X_train = train_df.drop(['is_bot'], axis = 1) 
    y_train = train_df['is_bot']
    
    clf = MultinomialNB(alpha=1, fit_prior=False)
    clf.fit(X_train, y_train)
    
    pickle.dump(clf, open(model_name, 'wb'))
    
#create_clf_model("train_data.csv", "tfidf_model.sav")

"""

to include tfidf in the aggregated model--

get the train and test data and run it through the preprocessing function:

train_df = pd.read_csv("train_data.csv")
train_df = preprocess_df(train_df)

test_df = pd.read_csv("test_data.csv")
test_df = preprocess_df(test_df)

get X_train and X_test (by dropping the label):

X_train = train_df.drop(['is_bot'], axis = 1) 

X_test = test_df.drop(['is_bot'], axis = 1) 

load the tfidf classifier:

clf = pickle.load(open('tfidf_model.sav', 'rb'))

get prediction labels using the .predict method:

y_test_pred = clf.predict(X_test)
y_train_pred = clf.predict(X_train)

"""
