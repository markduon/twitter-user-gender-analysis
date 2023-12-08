import pickle
import pandas as pd
from gensim.summarization import summarize
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score


def lowercase_data(data):
    data = data.lower()
    return data

def clear_data(dataframe):
    data = pd.DataFrame(dataframe.apply(lambda x: lowercase_data(x)))
    data.replace('[@+]', "", regex=True,inplace=True)
    data.replace('[()]', "", regex=True,inplace=True)
    data.replace('[#+]', "", regex=True,inplace=True)
    url_regex = '''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
    data = data.replace(url_regex, "", regex=True)
    return data

def predict_data(data):
    df_test = pd.read_csv(data, encoding='latin1')
    col = ['Unnamed: 0', 'is_bot', 'description', 'text']
    df_test = df_test[col]
    df_test["description"].fillna("", inplace = True)
    df_test["content"] = df_test["description"]  + ". " + df_test["text"]
    data_content_test = clear_data(dataframe=df_test["content"])
    data_test = pd.concat([df_test["Unnamed: 0"],df_test["is_bot"],data_content_test["content"]],axis=1)
    
    # For tweets that cannot be summarized, just keep the original tweets
    label_test_list = []
    data_test_summarize_dict = {}
    for i, row in data_test.iterrows():
        id_value = row["Unnamed: 0"]
        text_value = row["content"]
        label = row['is_bot']
        try:
            data_summarize = summarize(text_value)
            if data_summarize == "" or data_summarize == " ":
                data_summarize = text_value
            data_test_summarize_dict[id_value] = data_summarize
            label_test_list.append(label)
        except:
            data_test_summarize_dict[id_value] = text_value
            label_test_list.append(label)
            continue

    data_test = pd.DataFrame(list(data_test_summarize_dict.items()), columns=['id', 'content'])
    data_test["label"] = label_test_list

    # Tokenization
    data_test['content'] = [nltk.word_tokenize(tweet) for tweet in data_test['content']]
    test_token = []
    for each_row in data_test['content']:
        test_token.append([i for i in each_row if i.isalpha()])

    # Remove stopwords
    test_remove_stop = []
    stop_words = set(stopwords.words('english'))
    for each_row in test_token:
        test_remove_stop.append([i for i in each_row if i not in stop_words])

    # Lemmatization
    lemma = nltk.WordNetLemmatizer()
    test_lemma = []
    for each_row in test_remove_stop:
        test_lemma.append([lemma.lemmatize(word) for word in each_row])
    
    data_test['content'] = test_lemma
    corpus_test = [" ".join(desc) for desc in data_test['content'].values]

    # Convert text to vector
    vectorizer = CountVectorizer(max_features = 1500, stop_words = "english")
    X_test = vectorizer.fit_transform(corpus_test).toarray()
    # df_vector_test = pd.DataFrame(X_test, columns=vectorizer.get_feature_names_out())
    y_test = data_test['label'].values
    return X_test, y_test



if __name__ == "__main__":

    X_test, y_test = predict_data(data="data/test_data.csv")
    with open('mlp_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy)

    # df_test = pd.read_csv("test_data.csv", encoding='latin1')
    # col = ['Unnamed: 0', 'is_bot', 'description', 'text']
    # df_test = df_test[col]
    # df_test["description"].fillna("", inplace = True)
    # df_test["content"] = df_test["description"]  + ". " + df_test["text"]
    # data_content_test = clear_data(dataframe=df_test["content"])
    # data_test = pd.concat([df_test["Unnamed: 0"],df_test["is_bot"],data_content_test["content"]],axis=1)
    # label_test_list = []
    # data_test_summarize_dict = {}
    # for i, row in data_test.iterrows():
    #     id_value = row["Unnamed: 0"]
    #     text_value = row["content"]
    #     label = row['is_bot']
    #     try:
    #         data_summarize = summarize(text_value)
    #         if data_summarize == "" or data_summarize == " ":
    #             data_summarize = text_value
    #         data_test_summarize_dict[id_value] = data_summarize
    #         label_test_list.append(label)
    #     except:
    #         data_test_summarize_dict[id_value] = text_value
    #         label_test_list.append(label)
    #         continue

    # data_test = pd.DataFrame(list(data_test_summarize_dict.items()), columns=['id', 'content'])
    # data_test["label"] = label_test_list

    # data_test['content'] = [nltk.word_tokenize(tweet) for tweet in data_test['content']]
    # test_token = []
    # for each_row in data_test['content']:
    #     test_token.append([i for i in each_row if i.isalpha()])

    # test_remove_stop = []
    # # now remove them from the list
    # stop_words = set(stopwords.words('english'))
    # for each_row in test_token:
    #     test_remove_stop.append([i for i in each_row if i not in stop_words])

    # lemma = nltk.WordNetLemmatizer()
    # test_lemma = []
    # for each_row in test_remove_stop:
    #     test_lemma.append([lemma.lemmatize(word) for word in each_row])
    
    # data_test['content'] = test_lemma
    # corpus_test = [" ".join(desc) for desc in data_test['content'].values]

    # vectorizer = CountVectorizer(max_features = 1500, stop_words = "english")
    # X_test = vectorizer.fit_transform(corpus_test).toarray()
    # df_vector_test = pd.DataFrame(X_test, columns=vectorizer.get_feature_names_out())
    # y_test = data_test['label'].values

    # with open('mlp_model.pkl', 'rb') as f:
    #     clf = pickle.load(f)

    # y_pred = clf.predict(X_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy:.2f}")