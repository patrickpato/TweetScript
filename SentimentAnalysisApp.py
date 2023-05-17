import sys
import spacy 
import tweepy
from tweepy import OAuthHandler
from tweepy import API
import sqlite3
import argparse
import time
import pandas as pd
import psycopg2
import re
import joblib
import datetime as dt
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np
import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc
import plotly.express as px 
import sqlite3

nlp = spacy.load("en_core_web_sm")
client = tweepy.Client(bearer_token="AAAAAAAAAAAAAAAAAAAAACdPlAEAAAAAVMGdBjOH3o8Esyo6rIME1RwGHbg%3DYpI0kqwzzKAYUwVxFFTEBn70H9E6W0iW6KwaFb629Fug0txtIs")

def fetch_tweets(query):
    tweets = tweepy.Paginator(client.search_recent_tweets, query=query, tweet_fields=['author_id','created_at', 'text','source','public_metrics'],max_results=100).flatten(limit=10000)
    users, text, dates, metrics = [], [], [], []
    for tweet in tweets:
        users.append(tweet.author_id)
        text.append(tweet.text)
        dates.append(tweet.created_at)
        metrics.append(tweet.public_metrics)

    replies_count = [item['reply_count'] for item in metrics]
    impressions_counts = [item['impression_count'] for item in metrics]

    raw_tweets_df = pd.DataFrame(columns=["Date", "User_ID", "Tweet", "RepliesCount", "NumberofViews"])
    raw_tweets_df['Date'] = dates
    raw_tweets_df['User_ID'] = users
    raw_tweets_df['Tweet'] = text
    raw_tweets_df['RepliesCount'] = replies_count
    raw_tweets_df['NumberofViews'] = impressions_counts

    return raw_tweets_df

politics_df = fetch_tweets(["Raila", "Ruto"])
banks_df = fetch_tweets(["KCBGroup", "AbsaKenya"])
telkos_df = fetch_tweets(["Safaricom_Care", "AIRTEL_KE"])

#Cleaning the tweets using regex expressions
def clean_data(text):
    text = re.sub("@[A-Za-z0-9_]+","",text) #Remove @ sign
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text) #Remove http links
    #text = re.sub('\W',  " ", text)
    text = " ".join(text.split())
    #comment = ''.join(c for c in comment if c not in emoji.UNICODE_EMOJI) #Remove Emojis
    text = text.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    text = text.lower()
    return text
for df in [politics_df,  banks_df, telkos_df]:
    df['CleanTweets'] = df['Tweet'].apply(lambda x: clean_data(str(x)))
    
    
clf_pipeline = joblib.load("clf_pipeline.sav")
for df in [politics_df,  banks_df, telkos_df]:
    df['Predictions'] = clf_pipeline.predict(df['CleanTweets'])
    
for df in [politics_df,  banks_df, telkos_df]:
    df['B1Polarity'] = df['Predictions'].str.split("_")[:].str[0]
    df['B2Polarity'] = df['Predictions'].str.split("_")[:].str[1]
    
def identify_brand_aspects(df):
    #Part of Speech Aspect Tagging
    brand_aspects = dict()
    docs = nlp(str(df['CleanTweets']))
    for chunk in docs.noun_chunks:
        adj = " "
        noun = ""
        for tok in chunk:
            if tok.pos_ == "NOUN":
                noun = tok.text
            if tok.pos_ == "ADJ":
                adj=tok.text
        if noun:
            brand_aspects.update({noun:adj})
    return brand_aspects

#making statistical inferences from the predictions
def get_inference_stats(df):
        b1_pos = len(df[df['B1Polarity'] == "pos"])/len(df)
        b1_neg = len(df[df['B1Polarity'] == "neg"])/len(df)
        b2_pos = len(df[df['B2Polarity'] == "pos"])/len(df)
        b2_neg = len(df[df['B2Polarity'] == "neg"])/len(df)
        brand1_pos_tweets = df[df['B1Polarity'] == "pos"]
        brand2_pos_tweets = df[df['B2Polarity'] == "pos"]
        brand1_neg_tweets = df[df['B1Polarity'] == "neg"]
        brand2_neg_tweets = df[df['B2Polarity'] == "neg"]
        pos1_aspects = identify_brand_aspects(brand1_pos_tweets)
        pos2_aspects = identify_brand_aspects(brand2_pos_tweets)
        neg1_aspects = identify_brand_aspects(brand1_neg_tweets)
        neg2_aspects = identify_brand_aspects(brand2_neg_tweets)
        common_pos_aspects = [set(pos1_aspects.items()) & set(pos2_aspects.items())]
        common_neg_aspects = [set(neg1_aspects.items())& set(neg2_aspects.items())]
        if b1_pos > b2_pos:
            preferred_brand = "Brand1"
        elif b2_pos > b1_pos:
            preferred_brand = "Brand2"
        most_viral_tweet = df[df['RepliesCount']==df['RepliesCount'].max()]['Tweet'].values[0]
        most_viewed_tweet = df[df['NumberofViews'] == df['NumberofViews'].max()]['Tweet'].values[0]
        date = [dt.datetime.today().strftime("%m/%d/%Y")]
        results_df = pd.DataFrame(columns = ['Date', "Brand1Positivity", "Brand1Negativity","Brand2Positivity","Brand2Negativity","MostViralTweet", "MostViewedTweet", "PreferredBrand", "PosAspects", "NegAspects"])
        results_df['Date'] = date
        results_df["Brand1Positivity"] = b1_pos
        results_df["Brand1Negativity"] = b1_neg
        results_df["Brand2Positivity"] = b2_pos
        results_df["Brand2Negativity"] = b2_neg
        results_df['MostViralTweet'] = most_viral_tweet
        results_df['MostViewedTweet'] = most_viewed_tweet
        results_df['PreferredBrand'] = preferred_brand
        results_df['PosAspects'] = str(common_pos_aspects)
        results_df['NegAspects'] = str(common_neg_aspects)
        #print("Inferences made successfully")
        return results_df
politics_inferences= get_inference_stats(politics_df)
telkos_inferences = get_inference_stats(telkos_df)
banks_inferences = get_inference_stats(banks_df)

conn = sqlite3.connect("sample_web_app_database") 
c = conn.cursor()

c.execute("CREATE TABLE IF NOT EXISTS politicstrends( Date date, Brand1Positivity double precision, Brand2Positivity double precision, Brand1Negativity double precision, Brand2Negativity double precision, MostViralTweet text, MostViewedTweet text, PreferredBrand text, PosAspects text, NegAspects text)")
c.execute("CREATE TABLE IF NOT EXISTS telkostrends( Date date, Brand1Positivity double precision, Brand2Positivity double precision, Brand1Negativity double precision, Brand2Negativity double precision, MostViralTweet text, MostViewedTweet text, PreferredBrand text, PosAspects text, NegAspects text)")
c.execute("CREATE TABLE IF NOT EXISTS bankingtrends( Date date, Brand1Positivity double precision, Brand2Positivity double precision, Brand1Negativity double precision, Brand2Negativity double precision, MostViralTweet text, MostViewedTweet text, PreferredBrand text, PosAspects text, NegAspects text)")
conn.commit()

politics_inferences.to_sql("politicstrends", conn, if_exists="append", index=False)
telkos_inferences.to_sql("telkostrends", conn, if_exists="append", index=False)
banks_inferences.to_sql("bankingtrends", conn, if_exists="append", index=False)

'''
engine = create_engine("postgresql://postgres:paulowekulo@localhost:5432/brandcomparison")
politics_inferences.to_sql("politics_inferences", engine, if_exists="append")
banks_inferences.to_sql("banks_inferences", engine, if_exists="append")
telkos_inferences.to_sql("telkos_inferences", engine, if_exists="append")
'''
politics_archived = pd.read_sql("SELECT * FROM politicstrends", conn)
politics_archived_df = pd.DataFrame(politics_archived, columns = ["Date", "Brand1Positivity", "Brand2Positivity double", "Brand1Negativity", "Brand2Negativity", "MostViralTweet", "MostViewedTweet", "PreferredBrand", "PosAspects", "NegAspects"])
telkos_archived = pd.read_sql("SELECT * FROM telkostrends", conn) 
telkos_archived_df = pd.DataFrame(telkos_archived, columns = ["Date", "Brand1Positivity", "Brand2Positivity double", "Brand1Negativity", "Brand2Negativity", "MostViralTweet", "MostViewedTweet", "PreferredBrand", "PosAspects", "NegAspects"])
banking_archived = pd.read_sql("SELECT * FROM bankingtrends", conn)
banking_archived_df = pd.DataFrame(banking_archived, columns = ["Date", "Brand1Positivity", "Brand2Positivity double", "Brand1Negativity", "Brand2Negativity", "MostViralTweet", "MostViewedTweet", "PreferredBrand", "PosAspects", "NegAspects"])
print(banking_archived_df)