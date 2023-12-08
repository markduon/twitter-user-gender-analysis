from gensim.summarization import summarize

import random
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt     


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def lowercase_data(data):
    """Lowercase text

    Args:
        data

    Returns:
        data: dataframe that has lowercase text
    """
    data = data.lower()
    return data

def clear_data(dataframe):
    """Remove special characters from text

    Args:
        dataframe

    Returns:
        data: dataframe that having special characters to be removed
    """
    data = pd.DataFrame(dataframe.apply(lambda x: lowercase_data(x)))
    data.replace('[@+]', "", regex=True,inplace=True)
    data.replace('[()]', "", regex=True,inplace=True)
    data.replace('[#+]', "", regex=True,inplace=True)
    url_regex = '''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''
    data = data.replace(url_regex, "", regex=True)
    return data


if __name__ == "__main__":
    random.seed(42)
    # Load dataset
    df_train = pd.read_csv("data/train_data.csv", encoding='latin1') #2867 null
    df_test = pd.read_csv("data/test_data.csv", encoding='latin1') # 319 null

    # choose columns
    col = ['Unnamed: 0', 'is_bot', 'description', 'text']
    df_train = df_train[col]
    df_test = df_test[col]
    
    # Fill nan values in description column with empty string
    df_train["description"].fillna("", inplace = True)
    df_test["description"].fillna("", inplace = True)
    
    # Merge 2 columns "description" and "text" into 1 column "content"
    df_train["content"] = df_train["description"]  + ". " + df_train["text"]
    df_test["content"] = df_test["description"]  + ". " + df_test["text"]
    
    # Remove special characters
    data_content_train = clear_data(dataframe=df_train["content"])
    data_content_test = clear_data(dataframe=df_test["content"])

    # Concat 3 columns used in final dataframe
    data_train = pd.concat([df_train["Unnamed: 0"],df_train["is_bot"],data_content_train["content"]],axis=1)
    data_test = pd.concat([df_test["Unnamed: 0"],df_test["is_bot"],data_content_test["content"]],axis=1)
    

    # Text summarization
    count_train = 0

    label_train_list = []
    data_train_summarize_dict = {}
    for i, row in data_train.iterrows():
        id_value = row["Unnamed: 0"]
        text_value = row["content"]
        label = row['is_bot']
        try:
            data_summarize = summarize(text_value)
            if data_summarize == "" or data_summarize == " ":
                data_summarize = text_value
            data_train_summarize_dict[id_value] = data_summarize
            label_train_list.append(label)
        except:
            count_train += 1
            data_train_summarize_dict[id_value] = text_value
            label_train_list.append(label)
            continue
    
    count_test = 0

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
            count_test += 1
            data_test_summarize_dict[id_value] = text_value
            label_test_list.append(label)
            continue

    print("Total train samples cannot be summarized:",count_train)
    print("Total test samples cannot be summarized:",count_test)

    # Assign labels to the dataframe
    data_train = pd.DataFrame(list(data_train_summarize_dict.items()), columns=['id', 'content'])
    data_train["label"] = label_train_list
    
    data_test = pd.DataFrame(list(data_test_summarize_dict.items()), columns=['id', 'content'])
    data_test["label"] = label_test_list
    
    # Tokenize
    data_train['content'] = [nltk.word_tokenize(tweet) for tweet in data_train['content']]
    train_token = []
    for each_row in data_train['content']:
        train_token.append([i for i in each_row if i.isalpha()])

    data_test['content'] = [nltk.word_tokenize(tweet) for tweet in data_test['content']]
    test_token = []
    for each_row in data_test['content']:
        test_token.append([i for i in each_row if i.isalpha()])


    # Stopwords Removal
    train_remove_stop = []
    stop_words = set(stopwords.words('english'))
    # now remove them from the list
    for each_row in train_token:
        train_remove_stop.append([i for i in each_row if i not in stop_words])

    test_remove_stop = []
    # now remove them from the list
    for each_row in test_token:
        test_remove_stop.append([i for i in each_row if i not in stop_words])
    


    # Lemmatization
    train_lemma = []
    lemma = nltk.WordNetLemmatizer()
    for each_row in train_remove_stop:
        train_lemma.append([lemma.lemmatize(word) for word in each_row])
    
    test_lemma = []
    for each_row in test_remove_stop:
        test_lemma.append([lemma.lemmatize(word) for word in each_row])


    # put back the new sentences
    data_train['content'] = train_lemma
    data_test['content'] = test_lemma

    corpus_train = [" ".join(desc) for desc in data_train['content'].values]
    corpus_test = [" ".join(desc) for desc in data_test['content'].values]
    
    # Vectorize the text data
    vectorizer = CountVectorizer(max_features = 1500, stop_words = "english")
    X_train = vectorizer.fit_transform(corpus_train).toarray()
    X_test = vectorizer.fit_transform(corpus_test).toarray()

    y_train = data_train['label'].values
    y_test = data_test['label'].values

    # Using MLP to train model
    clf = MLPClassifier(hidden_layer_sizes=(50,),random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    # # with open("mlp_model.pkl", 'wb') as model_file:
    # #     pickle.dump(clf, model_file)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(conf_matrix)

    report = classification_report(y_test, y_pred)
    print("Random Forest Classification Report:\n", report)

    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1score = f1_score(y_test, y_pred, average='macro')

    print('Accuracy: %.4f \n' % accuracy)

    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    print(f"F1 Score = {f1score}")

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['human','bot'])
    disp.plot()
    plt.show()
