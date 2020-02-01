import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

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

df_count_by_classification_t