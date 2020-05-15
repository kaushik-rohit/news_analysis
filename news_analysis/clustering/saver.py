import argparse
import calendar
import multiprocessing as mp
from datetime import date, timedelta

import pandas as pd
from gensim import models, corpora

from clustering.cluster_analysis import get_similar_articles, get_articles_not_in_cluster
from shared import db

# create necessary arguments to run the analysis
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--db-path',
                    type=str,
                    required=True,
                    help='the path to database where news articles are stored')

parser.add_argument('-m', '--month',
                    choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    type=int,
                    default=None,
                    help='The month for which analysis is to be performed. If month is not provided '
                         'the analysis is performed on the whole year')

parser.add_argument('-y', '--year',
                    choices=[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
                             2015, 2016, 2017, 2018],
                    type=int,
                    default=2014,
                    help='The year for which analysis is to be performed. By default takes the value 2014')

parser.add_argument('-dict', '--dictionary',
                    type=str,
                    required=True,
                    help='the path to bag of words model')

parser.add_argument('-tf', '--tfidf-model',
                    type=str,
                    required=True,
                    help='the path to trained tfidf gensim model')

parser.add_argument('-t', '--threshold',
                    type=float,
                    default=0.3,
                    help='threshold for cosine similarity to consider news articles in same cluster')


def get_cluster_for_the_day(curr_date, path, dct, tfidf_model, threshold):
    """
    """
    delta = timedelta(days=1)
    next_date = curr_date + delta
    rows = []
    conn = db.NewsDb(path)

    print('calculating clusters for {}'.format(curr_date))

    articles_day1 = list(conn.select_articles_by_date(curr_date))
    articles_day2 = list(conn.select_articles_by_date(next_date))

    conn.close()
    unclustered_articles_indices = get_articles_not_in_cluster(articles_day1, dct, tfidf_model, threshold=threshold)
    unclustered_articles = [articles_day1[i] for i in unclustered_articles_indices]
    clustered_articles = [articles_day1[i] for i in range(len(articles_day1)) if i not in unclustered_articles_indices]
    unclustered_articles_indices_in_day2_cluster = get_similar_articles(unclustered_articles, articles_day2, dct,
                                                                        tfidf_model, threshold=threshold)

    rows_in_cluster = []
    rows_not_in_cluster = []
    rows_in_tomorrows_cluster = []

    for article in clustered_articles:
        rows_in_cluster += [[article.date, article.source_id, article.source, 'in_cluster', article.program_name]]

    for article in unclustered_articles:
        rows_not_in_cluster += [
            [article.date, article.source_id, article.source, 'not_in_cluster', article.program_name]]

    for i, indices in unclustered_articles_indices_in_day2_cluster:
        article = unclustered_articles[i]

        for idx in indices:
            tomorrow_article = articles_day2[idx]
            rows_in_tomorrows_cluster += [[article.date, article.source_id, article.source, 'in_tomorrows_cluster',
                                           article.program_name, article.transcript, tomorrow_article.source,
                                           tomorrow_article.program_name, tomorrow_article.transcript]]

    return rows_in_cluster, rows_not_in_cluster, rows_in_tomorrows_cluster


def save_clusters_for_month(db_path, dct, tfidf_model, year, month, threshold):
    """
    """
    assert (1 > threshold > 0)
    start_date = date(year, month, 1)
    end_date = date(year, month, calendar.monthrange(year, month)[1])  # calendar.monthrange(year, month)[1]

    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    print('calculating clusters for {} {}'.format(year, calendar.month_name[month]))
    pool = mp.Pool(mp.cpu_count())  # calculate stats for date in parallel
    stats = pool.starmap(get_cluster_for_the_day, [(curr_date, db_path, dct, tfidf_model, threshold)
                                                   for curr_date in date_range])
    pool.close()

    rows_in_cluster = []
    rows_not_in_cluster = []
    rows_in_tomorrows_cluster = []

    for stat in stats:
        rows_in_cluster += stat[0]
        rows_not_in_cluster += stat[1]
        rows_in_tomorrows_cluster += stat[2]

    in_cluster_df = pd.DataFrame(rows_in_cluster, columns=['date', 'source id', 'source_name', 'cluster', 'title'])
    in_cluster_df.to_csv(path_or_buf='../results/in_cluster_{}_{}.csv'.format(year, month))

    not_in_cluster_df = pd.DataFrame(rows_not_in_cluster, columns=['date', 'source id', 'source_name', 'cluster',
                                                                   'title'])
    not_in_cluster_df.to_csv(path_or_buf='../results/not_in_cluster_{}_{}.csv'.format(year, month))

    in_tomorrows_cluster_df = pd.DataFrame(rows_in_tomorrows_cluster, columns=['date', 'source id', 'source_name',
                                                                               'cluster', 'article title',
                                                                               'article transcript',
                                                                               'tomorrows article source',
                                                                               'tomorrows article title',
                                                                               'tomorrows articles transcript'])
    in_tomorrows_cluster_df.to_csv(path_or_buf='../results/in_tomorrow_cluster_{}_{}.csv'.format(year, month))


def main():
    args = parser.parse_args()

    dct = corpora.Dictionary.load(args.dictionary)
    tfidf_model = models.TfidfModel.load(args.tfidf_model)
    year = args.year
    month = args.month
    threshold = args.threshold
    db_path = args.db_path
    save_clusters_for_month(db_path, dct, tfidf_model, year, month, threshold)


if __name__ == '__main__':
    main()
