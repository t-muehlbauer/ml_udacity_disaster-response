import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
import nltk
import re
import os
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

nltk.download(['punkt', 'wordnet']);

def load_data(database_filepath):
    """
    Load data from an SQLite database and prepare it for training a machine learning model.

    Args:
        database_filepath (str): The path to the SQLite database file.

    Returns:
        tuple: A tuple containing three elements:
            - X (pandas.Series): Input messages.
            - y (pandas.DataFrame): Target labels.
            - category_names (Index): Names of the target categories.
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    
    # Remove 'child_alone' because there are only zeros
    df = df.drop(['child_alone'],axis=1)
    
    # seperate dataset to input variables (X) and target variable (Y)
    X = df.message
    y = df[df.columns[4:]]
    
    # Save category names in a variable
    category_names = y.columns
    
    return X, y, category_names


def tokenize(text):
    """
    Takes a text string as input and performs several text preprocessing steps on it including:
    - replacing URLs in the text with a placeholder
    - tokenizing the text into individual words
    - lemmatizing the words to their base form
    - converting the words to lowercase
 
    Arguments:
    text -- the input text string to be processed

    Returns:
    clean_tokens -- a list of cleaned tokens extracted from the input text string
    """

    # replace all urls wirth a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # extract urls from given text
    detected_urls = re.findall(url_regex, text)

    # replace all urls to 'url'
    for url in detected_urls:
        text = text.replace(url, "url")

    # extract the word tokens
    tokens = word_tokenize(text)

    # lemmatize to the stem of a word
    lemmatizer = WordNetLemmatizer()
    
    # list clean tokens
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    """
    Build a machine learning pipeline for classifying text messages into multiple categories.

    The pipeline includes:
    - CountVectorizer: Converts text messages into a numerical format by counting the occurrences of each word.
    - TfidfTransformer: Computes TF-IDF (term frequency-inverse document frequency) scores for the words.
    - MultiOutputClassifier with RandomForestClassifier: Uses decision trees to classify messages into categories.

    Returns:
        GridSearchCV: A grid search cross-validation object to find the best hyperparameters for the model.
    """
    # Define the pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, token_pattern=None)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # Ideal parameters (commented out)
    # parameters = {
    #     'clf__min_samples_split': [2], 
    #     'clf__n_estimators': [50], 
    #     'features__text_pipeline__vect__ngram_range': (1, 1),
    # }
    
    # Set new parameters to reduce the size of the model
    parameters = {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2],
    }
    
    # Create a GridSearchCV object
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=4, verbose=2, cv=3)
    
    return model



def evaluate_model(model, X_test, y_test, category_names):
    """
    Evaluate the performance of a machine learning model.

    Args:
        model (object): The trained machine learning model.
        X_test (pandas.Series): Input messages for testing.
        y_test (pandas.DataFrame): True labels for testing.
        category_names (Index): Names of the target categories.

    Returns:
        None
    """
    y_pred = model.predict(X_test)
    # Uncomment the line below if you want to include a classification report
    # class_report = classification_report(y_test, y_pred, target_names=category_names)


def save_model(model, model_filepath):
    """
    Save a trained machine learning model to a file.

    Args:
        model (object): The trained machine learning model.
        model_filepath (str): The path where the model will be saved.

    Returns:
        None
    """
    # Save the improved trained model to a file
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()