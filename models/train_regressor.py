import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'omw-1.4'])

def load_data(database_filepath):
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM listings', engine)
    X = df.drop(['review_scores_rating'], axis=1)
    y = df.review_scores_rating

    return X, y

def tokenize(text):
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(X):

    text_feature = 'description'
    numeric_features = list(X.columns)
    numeric_features.remove(text_feature)

    preprocess = make_column_transformer(
        (Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ]), text_feature),
        (StandardScaler(), numeric_features)
    )

    pipeline = Pipeline([
        ('preprocess', preprocess),
        ('reg', DecisionTreeRegressor())
    ])

    parameters = {
        'preprocess__pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'preprocess__pipeline__vect__max_df': (0.5, 0.75, 1.0),
        'preprocess__pipeline__vect__max_features': (None, 5000, 10000),
        'preprocess__pipeline__tfidf__use_idf': (True, False),
        'reg': [ElasticNet(), DecisionTreeRegressor()]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=20)
    
    return model


def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print("The R2 score of the model is {}".format(r2))
    mse = mean_squared_error(y_test, y_pred)
    print("The mean squared error of the model is {}".format(mse))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X)
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the listings database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_regressor.py ../data/AirbnbRatings.db regressor.pkl')


if __name__ == '__main__':
    main()