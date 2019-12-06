from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from datetime import date, timedelta
from dateutil import parser as date_parser
import argparse
import pandas as pd
import numpy as np
import pickle
import os
from utils import *

#create necessary arguments to run the analysis
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--db_path', type=str, default='./news.db')
parser.add_argument('-s', '--start-date', type=str, default='1-1-2014')
parser.add_argument('-e', '--end-date', type=str, default='31-1-2014')
parser.add_argument('-v', '--vectorizer', type=str, default='None')
parser.add_argument('-a', '--aggregate', type=str, default='date')
parser.add_argument('-t', '--similarity-threshold', type=int, default=0.3)


def build_vectorizer(df):
    dflen = len(df)
    corpus = df['Transcript'].values
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer = vectorizer.fit(corpus)
    save(vectorizer, 'vec.pkl')

    return vectorizer

def get_similar_articles(articles1, articles2, vec, threshold=0.3):
    """
    @df1: (DataFrame) corpus of news articles
    @df2: (DataFrame) corpus of news articles
    @vec: TfIdf Vectorizer, trained on a corpus that is superset of df1 & df2

    returns: list of articles from articles1 that are similar to articles from 
             articles2 and have different news source

    """

    articles1_len = len(articles1)
    articles2_len = len(articles2)

    similar_articles = []
    tfidf = vec.transform(articles2['Transcript'].values)
    
    articles2 = articles2.reset_index()
    
    for i in range(articles1_len):
        row = articles1.iloc[i]
        t1 = row['Transcript']
        source = row['Source']

        #indices of articles(2) where source is different than ith article in articles1
        idx = articles2.index[articles2['Source'] != row['Source']]

        #cosine similarity of ith article from articles1 to all articles in articles2
        cosine_similarities = linear_kernel(vec.transform([t1]), tfidf).flatten()

        assert(len(cosine_similarities) == len(articles2))
        assert(max(idx) < len(cosine_similarities))

        #cosine similarity of ith article from articles1 to articles in articles2
        #whose source is not same as source of ith article
        cosine_similarities = cosine_similarities[idx]

        #check that there is atleast one element such that similarity of ith article
        #is more than 0.3, if so ith article is in cluster with atleast one article
        #from articles2
        if np.count_nonzero(cosine_similarities > threshold) > 0:
            similar_articles.append(row)

    return similar_articles


def get_articles_not_in_cluster(articles, vec, threshold=0.3):
    """
    @articles:  pandas dataframe containing rows as articles
    @vec:       trained TfidfVectorizer on articles
    @threshold: articles are considered similar if cosine similarity is 
                greater than threshold

    returns: dataframe with articles which are not similar or in same cluster
             with any other article in the corpus
    """

    assert(isinstance(articles, pd.DataFrame))

    articles_n = len(articles)
    indices_of_unclustered_articles = []
    corpus = articles['Transcript'].values
    tfidf = vec.transform(corpus) #transfor transcript corpus to vector

    for i in range(articles_n):
        article = articles.iloc[i]

        #find cosine similarites of article i, to every other article
        cosine_similarities = linear_kernel(tfidf[i], tfidf).flatten()

        #check if article i, has cosine similarity greater than threshold
        #for none of the articles other than itself, cosine_sim(x,x) = 1 > threshold
        if np.count_nonzero(cosine_similarities > threshold) <= 1:
            indices_of_unclustered_articles.append(i)

    return articles.iloc[indices_of_unclustered_articles]


def aggregate_by_date(articles, vec, start_date, end_date):
    """
    articles: DataFrame with news articles
    vec: The vectorizer trained on articles
    start_date: start date of the analysis
    end_date: end date of the analysis
    """

    delta = timedelta(days=1)
    curr_date = start_date

    stats = []
    
    while curr_date < end_date:
        next_date = curr_date + delta

        articles_for_today = articles[articles['Date'] == curr_date]
        articles_for_next_day = articles[articles['Date'] == next_date]

        assert(not articles_for_today.empty and not articles_for_next_day.empty)

        articles_unclustered = get_articles_not_in_cluster(articles_for_today, vec)
        articles_in_next_day_cluster = get_similar_articles(articles_unclustered, articles_for_next_day, vec)

        assert(articles_unclustered is not None and articles_in_next_day_cluster is not None)

        day_stat = [
             curr_date,
             len(articles_for_today),
             len(articles_unclustered),
             len(articles_in_next_day_cluster),
             round(len(articles_unclustered) / len(articles_for_today),2),
             round(len(articles_in_next_day_cluster) / len(articles_unclustered), 2)
        ]

        stats += [day_stat]

        print("date={}, "
              "total_articles={}, "
              "articles_not_in_cluster={}, "
              "articles_in_next_day_cluster={}, "
              "percent_of_articles_not_in_cluster={}, "
              "percent_of_articles_in_next_day_cluster={} ".format(*day_stat))

        curr_date += delta

    return pd.DataFrame(stats, columns=['source', 'total_articles', 'articles_not_in_cluster', 'articles_in_next_day_cluster', 'percent_of_articles_not_in_cluster', 'percent_of_articles_in_next_day_cluster'])


def aggregate_by_source(articles, vec, start_date, end_date):
    delta = timedelta(days=1)

    #get a set of sources
    sources = articles['Source'].unique()

    #the aggregated results (list of lists)
    #each list contains stats for unique news source
    #dimension are going to n*5 where n=number of unique news source
    #5 is the number of stats per news source

    stats = []

    for source in sources:
        articles_from_source = articles[articles['Source'] == source]
        articels_from_other_source  = articles[articles['Source'] != source]
        curr_date = start_date

        #stats for given source for each
        #the results are aggregated over day and added to variable stats
        day_stats = []

        while curr_date < end_date:
            next_date = curr_date + delta
            articles_for_today = articles_from_source[articles_from_source['Date'] == curr_date]
            articles_for_next_day = articles[articles['Date'] == next_date]

            assert(articles_for_today is not None and articles_for_next_day is not None)

            #articles for curr date with news source as source and not in any cluster
            articles_unclusterd = get_articles_not_in_cluster(articles_for_today, vec)
            articles_in_next_day_cluster = get_similar_articles(articles_unclustered, articles_for_next_day, vec)

            day_stats += [
                    len(articles_for_today),
                    len(articles_unclustered),
                    len(articles_in_next_day_cluster),
                    round(len(articles_unclustered) / len(articles_for_today),2),
                    round(len(articles_in_cluster_next_day) / len(articles_unclustered), 2)
                ]
        day_stats = np.array(day_stats)

        avg_stats = [source] #1st column must be source name
        avg_stats += day_stats.mean(axis=0) #take mean against column

        stats += [avg_stats]

        print("source={}, "
              "total_articles={}, "
              "articles_not_in_cluster={}, "
              "articles_in_next_day_cluster={}, "
              "percent_of_articles_not_in_cluster={}, "
              "percent_of_articles_in_next_day_cluster={} ".format(*avg_stats))
              
    return pd.DataFrame(stats, columns=['source', 'total_articles', 'articles_not_in_cluster', 'articles_in_next_day_cluster', 'percent_of_articles_not_in_cluster', 'percent_of_articles_in_next_day_cluster'])


def main():
    args = parser.parse_args()
    data = load_data(args.db_path)
    vectorizer = None
    if args.vectorizer == 'None':
        vectorizer = build_vectorizer(data)
    else:
        with open(args.vectorizer, 'rb') as f:
            vectorizer = pickle.load(f)

    if args.aggregate == 'date':
        aggregate_by_date(data, vectorizer, date(2014,1,1), date(2014,1,31))
    else:
        aggregate_by_source(data, vectorizer, date(2014,1,1), date(2014,1,31))


if __name__ == '__main__':
    main()
