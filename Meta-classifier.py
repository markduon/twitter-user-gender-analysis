import pandas as pd
import numpy as np
import nltk
nltk.download('words')
import warnings
warnings.simplefilter("ignore")
import pickle # to save the vectorizer and the trained model
from cleanTweet import cleanTweet
# from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import math
# from sklearn.ensemble import AdaBoostClassifier
from tfidf_aggregate_wonk import preprocess_df
from inference import predict_data
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


#preprocessing fuction for model2(HA/Yonex)
def preprocess_numerical(df_name):
    df = pd.read_csv(df_name, encoding="ISO-8859-1")

    p_df = df[['tweet_rate', 'retweet_rate', 'fav_rate']].copy()
    p_df['tweet_rate'] = p_df['tweet_rate'].apply(lambda x: math.log1p(x))
    # p_df['retweet_rate'] = p_df['retweet_rate'].apply(lambda x: math.log1p(x))
    p_df['fav_rate'] = p_df['fav_rate'].apply(lambda x: math.log1p(x))
    return p_df


#import models
model_1 = pickle.load(open('LR_model.pkl', 'rb')) # Huyen
vectorizer = pickle.load(open('bow.pkl', 'rb')) #Huyen
model_2 = pickle.load(open('classifier_numerical.pickle', 'rb')) # HA
model_3 = pickle.load(open('tfidf_model.sav', 'rb')) #cameron
model_4 = pickle.load(open('mlp_model.pkl', 'rb')) #vinh


# load the train, test data
train_df = pd.read_csv('data/train_data.csv', encoding='latin-1')
test_df = pd.read_csv('data/test_data.csv', encoding='latin-1')
test_df_copy = test_df.copy()
# train and test label
y_train_df = train_df['is_bot'].tolist()
y_test_df = test_df['is_bot'].tolist()


#preprocessing for model 1
# clean dataframe's tweet column
test_df['text'] = test_df['text'].apply(cleanTweet)
train_df['text'] = train_df['text'].apply(cleanTweet)

# Replace null values in "description" column with empty strings
test_df['description'].fillna('', inplace=True)
train_df['description'].fillna('', inplace=True)

# clean dataframe's tweet column
test_df['description'] = test_df['description'].apply(cleanTweet)
train_df['description'] = train_df['description'].apply(cleanTweet)

test_df['combined_text'] = test_df['description'] + ' ' + test_df['text']
train_df['combined_text'] = train_df['description'] + ' ' + train_df['text']

X_test = vectorizer.transform(test_df['combined_text'].tolist())
X_train= vectorizer.transform(train_df['combined_text'].tolist())

#preprocessing for model 2
X_train_model_2 = preprocess_numerical('data/train_data.csv')
X_test_model_2 = preprocess_numerical('data/test_data.csv')

#preprocess for model 3
train_df_model_3 = preprocess_df(train_df)
test_df_model_3 = preprocess_df(test_df)
X_train_model_3 = train_df_model_3.drop(['is_bot'], axis = 1)
X_test_model_3 = test_df_model_3.drop(['is_bot'], axis = 1)

#preprocessing for model 4
X_train_model_4, train_label_4 = predict_data('data/train_data.csv')
X_test_model_4, test_label_4 = predict_data('data/test_data.csv')

#Model 1 prediction
y_train_pred = model_1.predict(X_train)
y_test_pred = model_1.predict(X_test)

#model 2 prediction
y_train_pred_model_2 = model_2.predict(X_train_model_2)
y_test_pred_model_2 = model_2.predict(X_test_model_2)

#model 3 prediction
y_train_pred_model_3 = model_3.predict(X_train_model_3)
y_test_pred_model_3 = model_3.predict(X_test_model_3)

#model 4 prediction
y_train_pred_model_4 = model_4.predict(X_train_model_4)
y_test_pred_model_4 = model_4.predict(X_test_model_4)

#aggregate the model's predictions
stacked_data_train = np.column_stack((y_train_pred, y_train_pred_model_2, y_train_pred_model_3, y_train_pred_model_4))
stacked_data_test = np.column_stack((y_test_pred, y_test_pred_model_2, y_test_pred_model_3,y_test_pred_model_4))

#define the labels and features for meta classifier
meta_features_train, meta_labels_train = stacked_data_train, y_train_df
meta_features_test, meta_labels_test = stacked_data_test, y_test_df

#define the meta classifier
meta_classifier = DecisionTreeClassifier(random_state=42)

#fit the decision tree model for the training dataset
meta_classifier.fit(meta_features_train, meta_labels_train)
#predict the labels for the testing data
y_pred_test = meta_classifier.predict(meta_features_test)

print('Predict individual models accuracy:')
#individual accuracy
#model 1
accuracy_1 = accuracy_score(y_test_pred, meta_labels_test)
print(f"Accuracy of the Model 1: {accuracy_1:.2f}")

accuracy_2 = accuracy_score(y_test_pred_model_2, meta_labels_test)
print(f"Accuracy of the Model 2: {accuracy_2:.2f}")

accuracy_3 = accuracy_score(y_test_pred_model_3, meta_labels_test)
print(f"Accuracy of the Model 3: {accuracy_3:.2f}")

accuracy_4 = accuracy_score(y_test_pred_model_4, meta_labels_test)
print(f"Accuracy of the Model 4: {accuracy_4:.2f}")

# Calculate the accuracy of the meta-classifier on the training data
y_pred_train = meta_classifier.predict(meta_features_train)
accuracy_meta_classifier_train = accuracy_score(meta_labels_train, y_pred_train)
print(f"Accuracy of the Meta-Classifier on Training Data: {accuracy_meta_classifier_train:.6f}")

# Evaluate the accuracy of the meta-classifier
accuracy = accuracy_score(meta_labels_test, y_pred_test)
print(f"Accuracy of the meta-classifier on testing data: {accuracy:.6f}")
conf_plot = confusion_matrix(meta_labels_test, y_pred_test)
print(f"confusion plot for meta-classifier testing: {conf_plot}")

#calculate precision recall, f-1 score
report = classification_report(meta_labels_test, y_pred_test)
print("lassification report:")
print(report)

# Get feature importances (weights) for each feature (model's prediction)
feature_importances = meta_classifier.feature_importances_

# Print the feature importances
for i, importance in enumerate(feature_importances):
    print(f"Feature {i}: Importance = {importance:.4f}")

# Create a DataFrame with the predicted labels
predicted_labels_df = pd.DataFrame({'Predicted_Labels': y_pred_test})

# Concatenate the predicted labels DataFrame with the original test data
test_df_with_predictions = pd.concat([test_df_copy, predicted_labels_df], axis=1)

# Save the DataFrame with predicted labels as a CSV file
test_df_with_predictions.to_csv('test_data_with_predictions.csv', index=False)

print("Test data with predicted labels saved as 'test_data_with_predictions.csv'")

# comparison between the original and the final meta classifier predicted
# Load the data from CSV files
df_A = pd.read_csv('data/twitter_user_data.csv', encoding='latin-1') # load original file
df_B = pd.read_csv('test_data_with_predictions.csv', encoding='latin-1') # load the predicted file

# Merge the dataframes based on the row number
merged_df = pd.merge(df_A, df_B[['Unnamed: 0.1', 'is_bot', 'Predicted_Labels']], left_index=True, right_on='Unnamed: 0.1', how='inner')
# print(merged_df['gender'].value_counts())
# Define a mapping dictionary for 'gender' column
gender_mapping = {'male': 0, 'female': 0, 'brand': 1}

# Use the map function to apply the mapping to the 'gender' column
merged_df['gender'] = merged_df['gender'].map(gender_mapping)

# Select the desired columns
result_df_1 = merged_df[['name', 'gender', 'is_bot', 'Predicted_Labels']]

#save csv file for comparison
result_df_1.to_csv('Twitter_data_original_comparison.csv', index=False)
# print(result_df_1['gender'].value_counts())

labels_accuracy = accuracy_score(result_df_1['gender'], result_df_1['Predicted_Labels'])
print(f"Comparison between the original and the predicted: {labels_accuracy:.6f}")
# Create a confusion matrix for 'Predicted_Labels'
labels_confusion_matrix = confusion_matrix(result_df_1['gender'], result_df_1['Predicted_Labels'])
print(f"confusion plot for Comparison between the original and the predicted:: {labels_confusion_matrix}")