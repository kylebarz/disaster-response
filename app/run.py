import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Histogram
from sklearn.externals import joblib
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

#sql to load counts by classification type
sql_classification_counter = 'select sum(related) as Related, sum(request) as Request, sum(offer) as Offer, \
                                               sum(aid_related) as Aid_Related, sum(medical_help) as Medical_Help, sum(medical_products) as Medical_Products, \
                                               sum(search_and_rescue) as Search_And_Rescue, sum(security) as Security, sum(military) as Military, \
                                               sum(child_alone) as Child_Alone, sum(water) as Water, sum(food) as Food, sum(shelter) as Shelter, sum(clothing) as Clothing, \
                                               sum(money) as Money, sum(missing_people) as Missing_People, sum(refugees) as Refugees, sum(death) as Death, sum(other_aid) as Other_Aid, \
                                               sum(infrastructure_related) as Infrastructure_Related, sum(transport) as Transport, sum(buildings) as Buildings, \
                                               sum(electricity) as Electricity, sum(tools) as Tools, sum(hospitals) as Hospitals, sum(shops) as Shops, sum(aid_centers) as Aid_Centers, \
                                               sum(other_infrastructure) as Other_Infrastructure, sum(weather_related) as Weather_Related, sum(floods) as Floods, sum(storm) as Storm, \
                                               sum(fire) as Fire, sum(earthquake) as Earthquake, sum(cold) as Cold, sum(other_weather) as Other_Weather, sum(direct_report) as Direct_Report \
                                               from categorized_messages_tbl;'

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('categorized_messages_tbl', engine)
df_count_by_classification = pd.read_sql_query(sql_classification_counter, engine)
df_count_by_classification_t = df_count_by_classification.transpose()

# add column with length of messages
df['message_length']  = df['message'].str.len()

#add column with number of words in messages
#improvement opportunity: this would be better to add in the ETL pipeline so that it isn't executed on each start.
df['word_count'] = df['message'].apply(lambda x: len(tokenize(x)))

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    #show count by genre
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    #Show Count of Records by Classification
    classification_counts = []
    for x in df_count_by_classification_t.values:
        classification_counts.append(int(x))
    
    classification_names = list(df_count_by_classification_t.index)
    
    #Show Count of Words
    word_count = list(df['word_count'].values)
    
    # create visuals with plotly
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }, 
        {
            'data': [
                Bar(
                    x=classification_names,
                    y=classification_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Classifications',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Classification"
                }
            }
        },
        {
            'data': [
                Histogram(
                    x=word_count,
                    nbinsx=5
                )
            ],

            'layout': {
                'title': 'Histogram of Word Count',
                'yaxis': {
                    'title': "# of Messages"
                },
                'xaxis': {
                    'title': "Word Count"
                }
            }
        },
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
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )



def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()