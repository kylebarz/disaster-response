import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import pickle

nltk.download('wordnet')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score, make_scorer


def load_data(database_filepath):
    """ 
    Load data that was prepared by the process_data.py script. 
  
    Parameters: 
    database_filepath (string): Relative or Full Path to the SQLite Database.
  
    Returns: 
    X: Contains all of the messages.
    Y: Indicates 1 (True) or 0 (False) for all of the possible categories for historical messages.
    category_names: All of the possible categories for messages.

    """
    #X, Y, category_names = load_data(database_filepath)
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM categorized_messages_tbl', engine)
    
    X = df.filter(items=['id', 'message', 'original', 'genre'])

    #Drop Columns that Aren't Relevant for Predictions
    # - 'child_alone' has no responses
    y = df.drop(['id', 'message', 'original', 'genre', 'child_alone'], axis=1)

    #'Related' Column has several '2' values; updating these to 1
    y['related'] = y['related'].map(lambda x: 1 if x == 2 else x)
    
    return X['message'], y, y.columns.values


def tokenize(text):
    """ 
    Lemmatizes text passed to the function.
  
    Parameters: 
    text (string): Text to Lemmatize.
  
    Returns: 
    lemmatized_words: List of words contained in the text.

    """
    tokens = nltk.word_tokenize(text)
    lemmatizer = nltk.WordNetLemmatizer()
    
    lemmatized_words = []
    for word in tokens:
        lemmatized_words.append(lemmatizer.lemmatize(word).lower().strip())
        
    return lemmatized_words


def performance_metric(y_true, y_pred):
    f1_scores = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i])
        f1_scores.append(f1)
        
    score = np.median(f1_scores)
    return score


def build_model():
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
        'tfidf__use_idf':[True, False]
        #,'clf__estimator__n_estimators':[10, 20]
        #,'clf__estimator__min_samples_split':[2, 4]
    }
    
    pipelineV2 = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultiOutputClassifier(SVC()))])
    
    parametersV2 = {
        'tfidf__use_idf':[True, False]
        #,'clf__estimator__C':[0.1,10] 
        #,'clf__estimator__gamma':[0.1,0.01]
    }

    #scorer = make_scorer(performance_metric)
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 5)
    
    scorerV2 = make_scorer(performance_metric)
    cvV2 = GridSearchCV(pipelineV2, param_grid = parametersV2, \
                        scoring = scorerV2, verbose = 10)
    
    #return cvV2
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    y_test_prediction = model.predict(X_test)
    print(classification_report(Y_test.values, y_test_prediction, target_names=category_names))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
    
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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