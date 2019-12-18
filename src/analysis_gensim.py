from gensim import models
from gensim import corpora
from datetime import date, timedelta
import calendar
from gensim.test.utils import get_tmpfile
from gensim.similarities import Similarity
import multiprocessing as mp
import argparse
import numpy as np
import pandas as pd
import db
import dill
from utils import *
from models import *

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
                    default=0.3,
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

    # if we only want diff source articles cluster, we need to calculate at what indices
    # same source news occurs so that it similarities at these indices can be masked
    if diff_source:
        indices_of_same_source = get_pos_of_same_source_news(unclustered_articles, articles_day2)

    for idx, similarities in enumerate(index[unclustered_articles_vec]):
        # check that there is atleast one element such that similarity of ith article
        # is more than 0.3, if so ith article is in cluster with atleast one article
        # from articles2
        similarities = np.array(similarities)

        # mask all the similarities where articles have some source
        # since we only want to know unclustered articles forming cluster
        # from next day articles but different source
        if diff_source and len(indices_of_same_source[idx]) != 0:
            similarities[indices_of_same_source[idx]] = 0

        if np.count_nonzero(similarities > threshold) > 0:
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
    assert (1 > threshold > 0)

    indices = []
    index_tmpfile = get_tmpfile("index")

    index = Similarity(index_tmpfile, tfidf_model[iter(BoWIter(dct, corpus))], num_features=len(dct))

    for idx, similarities in enumerate(index):
        if np.count_nonzero(np.array(similarities) > threshold) <= 1:
            indices += [idx]

    return indices


def aggregate_by_year(path, dct, tfidf_model, year, threshold=0.3):
    """if source is None, aggregate by year aggregates the result of all the articles over the month,
    where if the source is not None, it aggregates the result of articles from source over month
    Parameters
    ----------
    @path: string, location to sqlite database
    @dct: gensim Dictionary object, bag of word model
    @tfidf_model: gensim tfidf model, pretrained
    @year: int, year of the month for which analysis is to be done
    @threshold (optional): float, similarity threshold

    Returns
    -------
    A panda Dataframe instance
    """
    assert (1 > threshold > 0)
    stats = []

    for month in range(1, 12 + 1):
        stats += [aggregate_by_month(path, dct, tfidf_model, year, month, avg=True, threshold=threshold)]

    stats = pd.concat(stats)
    stats_by_source = stats.drop(columns=['month']).groupby(['source']).mean()  # group by source
    stats_by_month = stats.drop(columns=['source']).groupby(['month']).sum()  # group by date

    # add percentages to stats
    stats_by_source['percent_of_unclustered_articles'] = (stats_by_source['unclustered_articles'] /
                                                          stats_by_source['total_articles']).fillna(0).round(2)
    stats_by_source['percent_of_articles_in_next_day_cluster'] = (
            stats_by_source['unclustered_articles_in_next_day_cluster'] /
            stats_by_source['unclustered_articles']).fillna(0).round(2)

    stats_by_month['percent_of_unclustered_articles'] = (stats_by_month['unclustered_articles'] /
                                                         stats_by_month['total_articles']).fillna(0).round(2)
    stats_by_month['percent_of_articles_in_next_day_cluster'] = (
            stats_by_month['unclustered_articles_in_next_day_cluster'] /
            stats_by_month['unclustered_articles']).fillna(0).round(2)

    return stats_by_source, stats_by_month


def aggregate_by_month(path, dct, tfidf_model, year, month, avg=False, threshold=0.3):
    """
    Parameters
    ----------
    @path: string, location to sqlite database
    @dct: gensim Dictionary object, bag of word model
    @tfidf_model: gensim tfidf model, pretrained
    @year: int, year of the month for which analysis is to be done
    @month: int
    @threshold (optional): float, similarity threshold

    Returns
    -------
    A panda Dataframe instance
    """
    assert (1 > threshold > 0)
    delta = timedelta(days=1)
    month_name = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun', 7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct',
                  11: 'nov', 12: 'dec'}
    start_date = date(year, month, 1)
    end_date = date(year, month, calendar.monthrange(year, month)[1])

    pool = mp.Pool(mp.cpu_count())  # calculate stats for date in parallel
    print('Parallelize on {} CPUs'.format(mp.cpu_count()))
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    stats = pool.starmap(get_stats_for_date, [(path, dct, tfidf_model, curr_date, threshold)
                                              for curr_date in date_range])
    pool.close()

    stats = pd.concat(stats)
    # average the stats for month, to be used by function aggregate by year, which reports results averaged by month
    if avg:
        stats = stats.drop(columns=['date'])
        stats['month'] = month_name[month]
        return stats

    stats_by_source = stats.drop(columns=['date']).groupby(['source']).mean().astype(int)  # group by source
    stats_by_date = stats.drop(columns=['source']).groupby(['date']).sum()  # group by date

    # add percentages to stats
    stats_by_source['percent_of_unclustered_articles'] = (stats_by_source['unclustered_articles'] /
                                                          stats_by_source['total_articles']).fillna(0).round(2)
    stats_by_source['percent_of_articles_in_next_day_cluster'] = (
            stats_by_source['unclustered_articles_in_next_day_cluster'] /
            stats_by_source['unclustered_articles']).fillna(0).round(2)

    stats_by_date['percent_of_unclustered_articles'] = (stats_by_date['unclustered_articles'] /
                                                        stats_by_date['total_articles']).fillna(0).round(2)
    stats_by_date['percent_of_articles_in_next_day_cluster'] = (
            stats_by_date['unclustered_articles_in_next_day_cluster'] /
            stats_by_date['unclustered_articles']).fillna(0).round(2)

    print(stats_by_date)
    print('-----------------')
    print(stats_by_source)
    return stats_by_source, stats_by_date


def initialize_results(counts):
    """
    Parameters
    ----------
    @counts:

    Returns
    -------
    A Dictionary
    """
    # source, year, month, day, total_articles, unclustered_articles, unclustered_articles_in_next_day_cluster
    results = {}
    for cnt in counts:
        results[cnt[0]] = [cnt[1], 0, 0]
    return results


def get_stats_for_date(path, dct_path, model_path, curr_date, threshold=0.3):
    """
    Parameters
    ----------
    @path: string, location to sqlite database
    @dct: gensim Dictionary object, bag of word model
    @tfidf_model: gensim tfidf model, pretrained
    @curr_date: datetime.date object
    @threshold (optional): float, similarity threshold

    Returns
    -------
    A panda Dataframe instance
    """
    assert (1 > threshold > 0)
    print('Calculating stats for the day: {}'.format(curr_date))
    delta = timedelta(days=1)
    next_date = curr_date + delta
    dct = corpora.Dictionary.load(dct_path)
    tfidf_model = models.TfidfModel.load(model_path)
    conn = db.ArticlesDb(path)

    articles_day1 = list(conn.select_articles_by_date(curr_date))
    articles_day2 = list(conn.select_articles_by_date(next_date))
    count_grouped_by_source = conn.get_count_of_articles_for_date_by_source(curr_date)
    assert (articles_day1 is not None and articles_day2 is not None)

    results = initialize_results(count_grouped_by_source)
    unclustered_articles_indices = get_articles_not_in_cluster(articles_day1, dct, tfidf_model, threshold=threshold)
    unclustered_articles = [articles_day1[i] for i in unclustered_articles_indices]
    unclustered_articles_indices_in_day2_cluster = get_similar_articles(unclustered_articles, articles_day2, dct,
                                                                        tfidf_model, threshold=threshold)
    unclustered_articles_in_day2_cluster = [unclustered_articles[i] for i in
                                            unclustered_articles_indices_in_day2_cluster]

    assert (unclustered_articles is not None and unclustered_articles_in_day2_cluster is not None)

    # update the count of unclustered articles for current date
    for itr in unclustered_articles:
        source = itr.get_source()
        results[source][1] += 1

    # update count of articles that form cluster in next day
    for it in unclustered_articles_in_day2_cluster:
        source = it.get_source()
        results[source][2] += 1

    conn.close()  # close database connection

    # modify dictionary to pandas dataframes, since it is easier to perform group operations
    df_rows = []
    for k in results:
        df_row = [k, curr_date]
        df_row += results[k]
        df_rows += [df_row]

    ret = pd.DataFrame(df_rows, columns=['source', 'date', 'total_articles', 'unclustered_articles',
                                         'unclustered_articles_in_next_day_cluster'])
    return ret


def main():
    args = parser.parse_args()

    dct = args.dictionary
    tfidf_model = args.tfidf_model
    year = args.year
    month = args.month
    threshold = args.threshold
    db_path = args.db_path

    if month is None:
        df1, df2 = aggregate_by_year(db_path, dct, tfidf_model, year, threshold=threshold)
        df1.to_csv(path_or_buf='../results/{}_source.csv'.format(year))
        df2.to_csv(path_or_buf='../results/{}_date.csv'.format(year))
    else:
        df1, df2 = aggregate_by_month(db_path, dct, tfidf_model, year, month, threshold=threshold)
        df1.to_csv(path_or_buf='../results/{}_{}_source.csv'.format(year, month))
        df2.to_csv(path_or_buf='../results/{}_{}_date.csv'.format(year, month))


if __name__ == '__main__':
    main()
