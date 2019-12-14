from gensim import models
from gensim import corpora
from datetime import date, timedelta
import calendar
from dateutil import parser as date_parser
from gensim.test.utils import get_tmpfile
from gensim.similarities import Similarity
import argparse
import numpy as np
import pickle
import os
from utils import *
from models import *


#create necessary arguments to run the analysis
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--db-path',
                    type=str,
                    required=True,
                    help='the path to database where news articles are stored')

parser.add_argument('-m', '--month',
                    choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    type=int,
                    default=None)

parser.add_argument('-y', '--year',
                    type=int,
                    default=2014)

parser.add_argument('-dict', '--dictionary',
                    type=str,
                    required=True,
                    help='')

parser.add_argument('-tf', '--tfidf-model',
                    type=str,
                    required=True,
                    help='')

parser.add_argument('-t', '--threshold',
                    type=float,
                    default=0.3)

parser.add_argument('-a', '--aggregate',
                    type=str,
                    choices=['date', 'source'],
                    default='date',
                    help='')


def get_pos_of_same_source_news(corpus1, corpus2):
    """
    The method calculates for each doc in corpus1, a list of indexes at which doc
    in corpus2 have same news source.

    Parameters
    ----------
    @corpus1: (DataFrame) corpus of news articles
    @corpus2: (DataFrame) corpus of news articles

    Returns
    -------
    a list of list containing indexes of corpus2, ith list contains indexes for
    doc[i] in corpus1 at which docs in corpus2 have same news source

    """
    corpus1_len = len(corpus1)
    corpus2_len = len(corpus2)

    same_source_indices = []

    for i in range(corpus1_len):
        same_source_index_for_doci = []
        for j in range(corpus2_len):
            if corpus1[i].get_source() == corpus2[j].get_source():
                same_source_index_for_doci += [j]
        same_source_indices += [same_source_index_for_doci]

    return np.array(same_source_indices)

def get_similar_articles(unclustered_articles, articles_day2, dct, tfidf_model, threshold=0.3, diff_source=True):
    """
    The method returns all the articles from articles1 which have atleast one
    article from articles2 where the cosine similarity is more than threshold.

    Parameters
    ----------
    @unclusterd_articles: list of Articles, which are not in any cluster
    @articles_day2: list of Articles, from next day
    @dct: A gensim Dictionary object, the bag of words model for corpus
    @tfidf_model: gensim tfidf model
    @threshold: int, threshold for similarity
    @diff_source: boolean, whether cluster from next day should be from same source
                  or different

    Returns
    -------
    list of articles from articles1 that are similar to articles from
             articles2 and have different news source

    """
    index_tmpfile = get_tmpfile("index")
    similar_articles = []
    index = Similarity(index_tmpfile, tfidf_model[iter(BoWIter(dct, articles_day2))], num_features=len(dct)) 
    unclustered_articles_vec = tfidf_model[iter(BoWIter(dct, unclustered_articles))]

    #if we only want diff source articles cluster, we need to calculate at what indices
    # same source news occurs so that it similarities at these indices can be masked
    if diff_source:
        indices_of_same_source = get_pos_of_same_source_news(unclustered_articles, articles_day2)

    for idx, similarities in enumerate(index[unclustered_articles_vec]):
        #check that there is atleast one element such that similarity of ith article
        #is more than 0.3, if so ith article is in cluster with atleast one article
        #from articles2
        similarities = np.array(similarities)

        #mask all the similarites where articles have some source
        #since we only want to know unclustered articles forming cluster
        #from next day articles but different source
        if diff_source:
            similarities[indices_of_same_source[idx]] = 0

        if np.count_nonzero(similarities >= threshold) > 0:
            similar_articles += [idx]

    return similar_articles


def get_articles_not_in_cluster(corpus, dct, tfidf_model, threshold=0.3):
    """This method returns all the article from the corpus input such that
    there is no other article in corpus with which it has cosine similarity
    greater than threshold.
    While comparing we use np.count_nonzero(cosine_similarities > threshold)<=1
    since there will always be an article x(itself) such that cosine_similarity
    of x with x is greater than threshold.

    Parameters
    ----------
    @corpus: list of Articles
    @dct: gensim Dictionary object, bag of word model
    @tfidf_model: gensim tfidf model, pretrained

    Returns
    -------
    a list of indexes at which articles in corpus do not form cluster with any
    other article
    """
    assert(threshold < 1 and threshold > 0)

    indices = []
    index_tmpfile = get_tmpfile("index")

    index = Similarity(index_tmpfile, tfidf_model[iter(BoWIter(dct, corpus))], num_features=len(dct))

    for idx, similarities in enumerate(index):
        if np.count_nonzero(np.array(similarities) >= threshold) <= 1:
            indices += [idx]

    return indices


def aggregate_by_source(path, dct, tfidf_model, year, month, threshold=0.3):
    pass

def aggregate_by_year(path, dct, tfidf_model, year, threshold=0.3):
    """
    Parameters
    ----------
    @path: string, location to sqlite database
    @dct: gensim Dictionary object, bag of word model
    @tfidf_model: gensim tfidif model, pretrained
    @year: int, year of the month for which analysis is to be done
    threshold (optional): float, similarity threshold

    Returns
    -------
    A panda Dataframe instance
    """
    assert(threshold < 1 and threshold > 0)

    month_name = {1:'jan', 2:'feb', 3:'mar', 4:'apr', 5:'may', 6:'jun', 7:'jul', 8:'aug', 9:'sep', 10:'oct', 11:'nov', 12:'dec'}
    stats = []
    columns = None

    for month in range(1, 12+1):
        stats_for_month = [month_name[month]]
        df = aggregate_by_month(path, dct, tfidf_model, year, month, threshold=threshold)
        columns = df.columns[1:].tolist()
        stats_for_month += df[columns].mean().tolist()
        stats_for_month[1] = int(stats_for_month[1])
        stats_for_month[2] = int(stats_for_month[2])
        stats_for_month[3] = int(stats_for_month[3])
        stats_for_month[4] = round(stats_for_month[4], 2)
        stats_for_month[5] = round(stats_for_month[5], 2)

        stats += [stats_for_month]

    columns = ['month'] + columns

    return pd.DataFrame(stats, columns=columns)

def aggregate_by_month(path, dct, tfidf_model, year, month, threshold=0.3):
    """
    Parameters
    ----------
    @path: string, location to sqlite database
    @dct: gensim Dictionary object, bag of word model
    @tfidf_model: gensim tfidif model, pretrained
    @year: int, year of the month for which analysis is to be done
    @month: int
    threshold (optional): float, similarity threshold

    Returns
    -------
    A panda Dataframe instance
    """
    assert(threshold < 1 and threshold > 0)
    delta = timedelta(days=1)

    start_date = date(year, month, 1)
    end_date = date(year, month, calendar.monthrange(year, month)[1])

    curr_date = start_date

    stats = []

    conn = db.articles_database(path)

    while curr_date < end_date:
        next_date = curr_date + delta

        count_day1 = conn.get_count_of_articles_for_date(curr_date)
        count_day2 = conn.get_count_of_articles_for_date(next_date)

        if count_day1 <=1 or count_day2 <= 1:
            curr_date += delta
            stats += [[curr_date, 0, 0, 0, 0, 0]]
            continue

        articles_day1 = list(iter(conn.select_articles_by_date(curr_date)))
        articles_day2 = list(iter(conn.select_articles_by_date(next_date)))

        unclustered_articles_indices = get_articles_not_in_cluster(articles_day1, dct, tfidf_model, threshold=threshold)
        unclustered_articles = [articles_day1[i] for i in unclustered_articles_indices]
        unclustered_articles_in_day2_cluster = get_similar_articles(unclustered_articles, articles_day2, dct, tfidf_model, threshold=threshold)

        assert(unclustered_articles is not None and unclustered_articles_in_day2_cluster is not None)

        day_stat = [
             curr_date,
             count_day1,
             len(unclustered_articles),
             len(unclustered_articles_in_day2_cluster),
             round(len(unclustered_articles) / len(articles_day1),2),
             round(len(unclustered_articles_in_day2_cluster) / len(unclustered_articles), 2)
        ]

        stats += [day_stat]

        print("date={}, "
              "total_articles={}, "
              "articles_not_in_cluster={}, "
              "articles_in_next_day_cluster={}, "
              "percent_of_articles_not_in_cluster={}, "
              "percent_of_articles_in_next_day_cluster={} ".format(*day_stat))

        curr_date += delta

    return pd.DataFrame(stats, columns=['date', 'total_articles', 'articles_not_in_cluster', 'articles_in_next_day_cluster', 'percent_of_articles_not_in_cluster', 'percent_of_articles_in_next_day_cluster'])


def main():
    args = parser.parse_args()

    dct = corpora.Dictionary.load(args.dictionary)
    tfidf_model = models.TfidfModel.load(args.tfidf_model)
    year = args.year
    month = args.month
    threshold = args.threshold
    db_path = args.db_path

    if args.aggregate == 'date':
        if month == None:
            res = aggregate_by_year(db_path, dct, tfidf_model, year, threshold=threshold)
            res.to_csv(path_or_buf='../results/{}_date.csv'.format(year))
        else:
            res = aggregate_by_month(db_path, dct, tfidf_model, year, month, threshold=threshold)
            res.to_csv(path_or_buf='../results/{}_{}_date.csv'.format(year, month))
    elif args.aggregate == 'source':
        pass


if __name__ == '__main__':
    main()
