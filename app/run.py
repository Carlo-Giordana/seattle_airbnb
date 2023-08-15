import json

import numpy as np
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
from plotly.graph_objs import Scatter
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/AirbnbRatings.db')
df = pd.read_sql_table('listings', engine)

# load model
model = joblib.load("../models/regressor.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    # First Chart - Pie Plot
    superhost_counts = df.groupby('host_is_superhost_t').count()
    superhost_values = list(superhost_counts)
    superhost_names = list(superhost_counts.index)

    # Second Chart - Bar Plot
    scores_counts = df.groupby('review_scores_rating').count()['description']
    scores_names = list(scores_counts.index)

    # create visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=superhost_names,
                    values=superhost_values
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=scores_names,
                    y=scores_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'automargin' : 'TRUE'
                },
                'margin': {
                    'b': 200
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    description = request.args.get('description', '')
    superhost = request.args.get('superhost', '')
    guests = request.args.get('guests', '')
    row = [
        description,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        1,
        1
    ]
    X_columns = df.drop('review_scores_rating', axis=1).columns
    X_new = pd.DataFrame([row], columns=X_columns, index=[0])

    # use model to predict classification for query
    predicted_rating = model.predict(X_new)[0]

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        description=description,
        predicted_rating=np.round(predicted_rating, 2)
    )


def main():
    app.run(host='localhost', port=3000, debug=True)


if __name__ == '__main__':
    main()