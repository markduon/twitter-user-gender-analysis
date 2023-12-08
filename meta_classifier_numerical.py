import os
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.exceptions import ConvergenceWarning
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import pickle


# Filter Data
def filter_data(df):
    df = df.drop(['description', 'fav_number', 'name', 'profileimage', 'retweet_count',
                  'text', 'tweet_count', 'date_created', 'date_last_judged', 'days_active'], axis=1)
    return df


# Feature Scaling
def feature_scaling(df):
    df['tweet_rate'] = df['tweet_rate'].apply(lambda x: math.log1p(x))
    # df['retweet_rate'] = df['retweet_rate'].apply(lambda x: math.log1p(x))
    df['fav_rate'] = df['fav_rate'].apply(lambda x: math.log1p(x))
    return df


# Shuffle Split Dataset
def split_train_valid_dataset(df, label_name):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=84)
    for train_index, valid_index in split.split(df, df[label_name]):
        train_set = df.iloc[train_index]
        valid_set = df.iloc[valid_index]

    X_train = train_set.drop(['is_bot', label_name], axis=1)
    y_train = train_set[[label_name]].values.ravel()
    # y_train_clabel = train_set[["is_bot"]].values.ravel()

    X_valid = valid_set.drop(['is_bot', label_name], axis=1)
    y_valid = valid_set[[label_name]].values.ravel()
    y_valid_clabel = valid_set[["is_bot"]].values.ravel()

    return X_train, y_train, X_valid, y_valid, y_valid_clabel


def split_test_dataset(df, label_name):
    X_test = df.drop(['is_bot', label_name], axis=1)
    y_test = df[[label_name]].values.ravel()
    y_test_clabel = df[["is_bot"]].values.ravel()

    return X_test, y_test, y_test_clabel


# Apply SMOTE to the training data
def smote_training_data(X_train, y_train):
    smote = SMOTE(random_state=84)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled


# Train the Model
def train_models(X, y, params, labels):
    mlps = []

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(batch_size=20, hidden_layer_sizes=(
            10, 2), random_state=84, max_iter=1000, **param)

        # some parameter combinations will not converge as can be seen on the
        # plots so they are ignored here
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=ConvergenceWarning, module="sklearn"
            )
            mlp.fit(X, y)

        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)

    return mlps


def plot_performance(mlps, labels, plot_args):
    fig, ax = plt.subplots()

    for mlp, label, args in zip(mlps, labels, plot_args):
        ax.plot(mlp.loss_curve_, label=label, **args)

    fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
    plt.show()


def get_best_model(mlps, X, y, params) -> MLPClassifier:
    scores_mlps = []
    for mlp in mlps:
        scores_mlps.append(mlp.score(X, y))
    print("Best training score: %f" % max(scores_mlps))
    index_max = scores_mlps.index(max(scores_mlps))
    print("Parameters with the best training score:\n{}".format(
        params[index_max]))
    return mlps[index_max]


# Validation
def validation_5fold(mlp, X_valid, y_valid):
    scores = cross_validate(mlp, X_valid, y_valid, scoring=(
        'accuracy', 'f1', 'precision', 'recall'), cv=5)
    print('Mean Accuracy\t: {}\tStd: {}'.format(
        scores['test_accuracy'].mean(), scores['test_accuracy'].std()))
    print(
        'Mean F1-Score\t: {}\tStd: {}'.format(scores['test_f1'].mean(), scores['test_f1'].std()))
    print('Mean Precision\t: {}\tStd: {}'.format(
        scores['test_precision'].mean(), scores['test_precision'].std()))
    print('Mean Recall\t: {}\tStd: {}'.format(
        scores['test_recall'].mean(), scores['test_recall'].std()))


# Comparison with the Contributor's Labels
def compare_contributor_label(y_valid_clabel, y_valid_pred):
    print("Comparison with Contributor's labels")
    comparison_result = metrics.confusion_matrix(
        y_valid_clabel, y_valid_pred).ravel()
    print("Both Models guess as non-bot\t\t: {} ({:.2f}%)".format(
        comparison_result[0], 100 * comparison_result[0]/sum(comparison_result)))
    print("Both Models guess as bot\t\t: {} ({:.2f}%)".format(
        comparison_result[3], 100 * comparison_result[3]/sum(comparison_result)))
    print("Only Our Model guesses as bot\t\t: {} ({:.2f}%)".format(
        comparison_result[1], 100 * comparison_result[1]/sum(comparison_result)))
    print("Only Our Model guesses as non-bot\t: {} ({:.2f}%)".format(
        comparison_result[2], 100 * comparison_result[2]/sum(comparison_result)))


# Test Set Performance
def test_model(mlp, X_test, y_test, y_test_clabel):
    y_test_pred = mlp.predict(X_test)

    print("Comparison with the Model's labels")
    print(metrics.classification_report(y_test, y_test_pred))

    # print("Comparison with the Ground Truth")
    # print(metrics.classification_report(y_ground_truth, y_test_pred))
    compare_contributor_label(y_test_clabel, y_test_pred)


# Save the Model
def save_model(mlp):
    name = "classifier_numerical.pickle"
    pickle.dump(mlp, open(name, 'wb'))


# Pre-processing Function
def preprocess_numerical(df_name):
    df = pd.read_csv(df_name, encoding="ISO-8859-1")

    p_df = df[['tweet_rate', 'retweet_rate', 'fav_rate']].copy()
    p_df['tweet_rate'] = p_df['tweet_rate'].apply(lambda x: math.log1p(x))
    # p_df['retweet_rate'] = p_df['retweet_rate'].apply(lambda x: math.log1p(x))
    p_df['fav_rate'] = p_df['fav_rate'].apply(lambda x: math.log1p(x))

    return p_df


# Wrap-up Training Function
def classifier_numerical(train_df, test_df, label_name):
    params = [
        {
            "solver": "sgd",
            "learning_rate": "constant",
            "momentum": 0,
            "learning_rate_init": 0.2,
        },
        {
            "solver": "sgd",
            "learning_rate": "constant",
            "momentum": 0.5,
            "nesterovs_momentum": False,
            "learning_rate_init": 0.2,
        },
        {
            "solver": "sgd",
            "learning_rate": "constant",
            "momentum": 0.5,
            "nesterovs_momentum": True,
            "learning_rate_init": 0.2,
        },
        {
            "solver": "sgd",
            "learning_rate": "invscaling",
            "momentum": 0,
            "learning_rate_init": 0.2,
        },
        {
            "solver": "sgd",
            "learning_rate": "invscaling",
            "momentum": 0.5,
            "nesterovs_momentum": True,
            "learning_rate_init": 0.2,
        },
        {
            "solver": "sgd",
            "learning_rate": "invscaling",
            "momentum": 0.5,
            "nesterovs_momentum": False,
            "learning_rate_init": 0.2,
        },
        {"solver": "adam", "learning_rate_init": 0.01},
    ]

    labels = [
        "constant learning-rate",
        "constant with momentum",
        "constant with Nesterov's momentum",
        "inv-scaling learning-rate",
        "inv-scaling with momentum",
        "inv-scaling with Nesterov's momentum",
        "adam",
    ]

    plot_args = [
        {"c": "red", "linestyle": "-"},
        {"c": "green", "linestyle": "-"},
        {"c": "blue", "linestyle": "-"},
        {"c": "red", "linestyle": "--"},
        {"c": "green", "linestyle": "--"},
        {"c": "blue", "linestyle": "--"},
        {"c": "black", "linestyle": "-"},
    ]

    train_df = filter_data(train_df)
    test_df = filter_data(test_df)

    s_train_df = feature_scaling(train_df)
    s_test_df = feature_scaling(test_df)

    X_train, y_train, X_valid, y_valid, y_valid_clabel = split_train_valid_dataset(
        s_train_df, label_name)
    X_test, y_test, y_test_clabel = split_test_dataset(
        s_test_df, label_name)

    X_train_resampled, y_train_resampled = smote_training_data(
        X_train, y_train)

    print("Training:")
    mlps = train_models(X_train_resampled, y_train_resampled, params, labels)
    plot_performance(mlps, labels, plot_args)
    mlp = get_best_model(mlps, X_train_resampled, y_train_resampled, params)

    print("\n\nValidation:")
    print("K-fold Validation (K=5):")
    validation_5fold(mlp, X_valid, y_valid)
    y_valid_pred = mlp.predict(X_valid)
    compare_contributor_label(y_valid_clabel, y_valid_pred)

    print("\n\nTest:")
    # test_model(mlp, X_test, y_test, y_test_clabel)
    y_test_pred = mlp.predict(X_test)
    print("Comparison with the Model's labels")
    print(metrics.classification_report(y_test, y_test_pred))
    compare_contributor_label(y_test_clabel, y_test_pred)

    # save_model(mlp)
    return mlp


# Main Function
train_df = pd.read_csv(
    './data/train_data_labeled_numeric.csv', encoding="ISO-8859-1")
train_df = train_df.set_index("X", verify_integrity=True)
train_df = train_df.drop(['kmeans_pred', 'X_golden'], axis=1)
test_df = pd.read_csv(
    './data/test_data_labeled_numeric.csv', encoding="ISO-8859-1")
test_df = test_df.set_index("X", verify_integrity=True)
test_df = test_df.drop(['Unnamed..0', 'kmeans_pred', 'X_golden'], axis=1)
classifier_numerical(train_df, test_df, 'SOM_pred')
# save_model(mlp)
