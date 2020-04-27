from datetime import date, timedelta
from gensim import models, corpora
from gensim.similarities import MatrixSimilarity
import multiprocessing as mp
import argparse
import numpy as np
import calendar
import pandas as pd
import db
from models import *
import helpers

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

parser.add_argument('--diff-source-incluster',
                    type=bool,
                    default=True,
                    help='if True consider only articles from different source to be in cluster for same date')

parser.add_argument('--diff-source-tomorrow-cluster',
                    type=bool,
                    default=True,
                    help='if True consider only articles from different source to be in cluster from next date')

parser.add_argument('-cm', '--cluster-median',
                    action='store_true',
                    help='')


def get_pos_of_same_source_news(corpus1, corpus2=None):
    """
    The method calculates for each doc in corpus1, a list of indexes at which doc
    in corpus2 have same news source.

    Parameters
    ----------
    @corpus1: (DataFrame) corpus of news articles
    @corpus2: (DataFrame) corpus of news articles

    Returns
    -------
    if corpus2 is not None:
    a list of list containing indexes of corpus2, ith list contains indexes for
    doc[i] in corpus1 at which docs in corpus2 have same news source
    else if corpus2 is None:
    a list of list containing indexes of corpus1, ith list contains indexes for
    doc[i] in corpus1 at which docs in corpus1 have same news source
    """
    if corpus2 is None:
        corpus2 = corpus1

    corpus1_len = len(corpus1)
    corpus2_len = len(corpus2)
    same_source_indices = []

    for i in range(corpus1_len):
        same_source_index_for_doci = []
        for j in range(corpus2_len):
            if corpus1[i].source == corpus2[j].source:
                same_source_index_for_doci += [j]
        same_source_indices += [same_source_index_for_doci]

    return np.array(same_source_indices)


def check_not_subset(indices1, indices2):
    """
    Check whether indices1 is subset of indices2, i.e there is atleast one element in indices1 that is not present
    in indices2
    Parameters
    ----------
    indices1: list of int
    indices2: list of int

    Returns
    -------
    True if atleast one element from indices1 is not part of indices2
    else False
    """
    for index in indices1:
        if index not in indices2:
            return True

    return False


def get_similarities(articles1, articles2, dct, tfidf_model):
    """
    Return similarity of each article from articles1 to each article from articles2
    :param articles1:
    :param articles2:
    :param dct:
    :param tfidf_model:
    :return:
    """
    similarity_ret = []
    filter_fn = helpers.preprocess_text
    index = MatrixSimilarity(tfidf_model[list(iter(BoWIter(dct, articles2, filter_fn)))], num_features=len(dct))
    articles1_vec = tfidf_model[iter(BoWIter(dct, articles1, filter_fn))]

    for idx, similarities in enumerate(index[articles1_vec]):
        similarities = np.array(similarities)
        similarity_ret.append((idx, similarities))
    return similarity_ret


def get_similar_articles(articles1, articles2, dct, tfidf_model, threshold=0.3, diff_source=True):
    """
    The method returns all the articles from articles1 which have atleast one
    article from articles2 where the cosine similarity is more than threshold.
    Moreover, if the parameter return_articles is True, the function also returns
    the indexes of articles from articles_day2 to which articles in unclustered articles
    are similar.

    Parameters
    ----------
    @unclusterd_articles: list of Articles, which are not in any cluster
    @articles_day2: list of Articles, from next day
    @dct: A gensim Dictionary object, the bag of words model for corpus
    @tfidf_model: gensim tfidf model
    @return_articles: boolean, whether indices of articles which article at index i is similar to will
                      also be returned
    @threshold: int, threshold for similarity
    @diff_source: boolean, whether cluster from next day should be from same source
                  or different

    Returns
    -------
    list of tuples, where the first element in the tuple is the index of articles from unclustered articles which have
    atleast one article from articles_day2 with greater similarity and the second element in the tuple is the index of
    articles from articles_day with which similarity threshold is greater

    """
    similar_articles = []
    filter_fn = helpers.preprocess_text
    index = MatrixSimilarity(tfidf_model[list(iter(BoWIter(dct, articles2, filter_fn)))], num_features=len(dct))
    articles1_vec = tfidf_model[iter(BoWIter(dct, articles1, filter_fn))]

    # if we only want diff source articles cluster, we need to calculate at what indices
    # same source news occurs so that it similarities at these indices can be masked
    if diff_source:
        indices_of_same_source = get_pos_of_same_source_news(articles1, articles2)

    for idx, similarities in enumerate(index[articles1_vec]):
        # check that there is atleast one element such that similarity of ith article
        # is more than 0.3, if so ith article is in cluster with atleast one article from articles2
        similarities = np.array(similarities)
        indices_where_similarity_greater_than_threshold = np.argwhere(similarities > threshold).flatten()

        # if similarity with no article is greater than 0, continue to next article
        if indices_where_similarity_greater_than_threshold.size == 0:
            continue

        # if diff_source is true, check that alteast one similar article is of different source
        if diff_source and len(indices_of_same_source[idx]) != 0:
            indices_of_same_source_i = indices_of_same_source[idx]

            assert (len(similarities) >= len(indices_of_same_source_i))
            assert (len(similarities) > max(indices_of_same_source_i))

            # check if atleast one similar article is from different source
            if not check_not_subset(indices_where_similarity_greater_than_threshold, indices_of_same_source_i):
                continue

        similar_articles += [(idx, list(indices_where_similarity_greater_than_threshold))]

    return similar_articles


def get_articles_in_cluster(corpus, dct, tfidf_model, threshold=0.3, diff_source=True):
    """
    Calculates similarity of articles from corpus with all the articles and return a list of indices of articles to
    which it is similar expect itself. If diff source is True then articles can only be similar to articles from
    different source.

    Parameters
    ----------
    @corpus: list of Articles
    @dct: gensim Dictionary object, bag of word model
    @tfidf_model: gensim tfidf model, pretrained
    @threshold: threshold for considering articles similar
    @diff_source: if similar articles are from different source

    Returns
    -------
    list of tuples, where the first element in the tuple is the index of articles from corpus which have
    atleast one article within corpus except itself with greater similarity and the second element in the tuple is the
     index of articles at which similarity threshold is greater
    """

    in_cluster = []
    filter_fn = helpers.preprocess_text
    index = MatrixSimilarity(tfidf_model[list(iter(BoWIter(dct, corpus, filter_fn)))], num_features=len(dct))
    # if we only want diff source articles cluster, we need to calculate at what indices
    # same source news occurs so that it similarities at these indices can be masked
    if diff_source:
        indices_of_same_source = get_pos_of_same_source_news(corpus)

    for idx, similarities in enumerate(index):

        similarities[idx] = 0  # mask similarity with itself
        indices_where_similarity_greater_than_threshold = np.argwhere(similarities > threshold).flatten()

        # if similarity with no article is greater than 0, continue to next article
        if indices_where_similarity_greater_than_threshold.size == 0:
            continue

        # mask all the similarities where articles have some source
        # since we only want to know unclustered articles forming cluster
        # from next day articles but different source
        if diff_source and len(indices_of_same_source[idx]) != 0:
            indices_of_same_source_i = indices_of_same_source[idx]

            assert (len(similarities) >= len(indices_of_same_source_i))
            assert (len(similarities) > max(indices_of_same_source_i))

            # check if atleast one similar article is from different source
            if not check_not_subset(indices_where_similarity_greater_than_threshold, indices_of_same_source_i):
                continue

        in_cluster += [(idx, list(indices_where_similarity_greater_than_threshold))]

    return in_cluster


def get_articles_not_in_cluster(corpus, dct, tfidf_model, threshold=0.3, diff_source=True):
    """This method returns all the article from the corpus input such that
    there is no other article in corpus with which it has cosine similarity
    greater than threshold.
    While comparing we use np.count_nonzero(cosine_similarities > threshold)<=1
    since there will always be an article x(itself) such that cosine_similarity
    of x with x is greater than threshold. ie cosine (x, x) = 1

    Parameters
    ----------
    @corpus: list of Articles
    @dct: gensim Dictionary object, bag of word model
    @tfidf_model: gensim tfidf model, pretrained

    Returns
    -------
    a list of indexes of articles in corpus which are not similar to any other articles
    """
    assert (1 > threshold > 0)

    indices = []
    filter_fn = helpers.preprocess_text
    index = MatrixSimilarity(tfidf_model[list(iter(BoWIter(dct, corpus, filter_fn)))], num_features=len(dct))

    # if we only want diff source articles cluster, we need to calculate at what indices
    # same source news occurs so that it similarities at these indices can be masked
    if diff_source:
        indices_of_same_source = get_pos_of_same_source_news(corpus)

    for idx, similarities in enumerate(index):

        # mask all the similarities where articles have some source
        # since we only want to know unclustered articles forming cluster
        # from next day articles but different source
        if diff_source and len(indices_of_same_source[idx]) != 0:
            indices_of_same_source_i = indices_of_same_source[idx]

            assert (len(similarities) >= len(indices_of_same_source_i))
            assert (len(similarities) > max(indices_of_same_source_i))

            similarities[indices_of_same_source_i] = 0

        similarities[idx] = 0  # mask similarity with itself

        if np.count_nonzero(np.array(similarities) > threshold) == 0:
            indices += [idx]

    return indices


def get_within_source_cluster_for_the_day(curr_date, path, dct, tfidf_model, threshold):
    """
    Parameters
    ----------
    path: (string) path to articles database
    dct: (gensim dictionary)
    tfidf_model: (gensim tfidf model)
    curr_date: (python datetime object) the date for which clusters of articles is to be calculated
    threshold: (float) the cosine similarity threshold which is used to classify articles into same cluster

    Returns
    -------
    clustered_articles: the list of articles for curr_date which are in cluster with other articles
    unclustered_articles: the list of articles for curr_date which are not in cluster with any other article
    unclustered_articles_in_day2_cluster: list of articles for curr_date which are not in cluster with any other article
    from same date but are in cluster with articles from next day
    """

    delta = timedelta(days=1)
    next_date = curr_date + delta
    conn = db.NewsDb(path)

    print('calculating within source clusters for {}'.format(curr_date))
    within_source_tomorrow_cluster = {source: [] for source in helpers.source_names}
    within_source_in_cluster = {source: [] for source in helpers.source_names}

    articles_day1 = list(conn.select_articles_by_date(curr_date))
    articles_day2 = list(conn.select_articles_by_date(next_date))

    conn.close()

    unclustered_articles_indices = get_articles_not_in_cluster(articles_day1, dct, tfidf_model, threshold=threshold)
    unclustered_articles = [articles_day1[i] for i in unclustered_articles_indices]
    unclustered_articles_indices_in_day2_cluster = get_similar_articles(unclustered_articles, articles_day2, dct,
                                                                        tfidf_model, threshold=threshold)

    for idx, indices in unclustered_articles_indices_in_day2_cluster:
        source = unclustered_articles[idx].source
        articles = [articles_day2[i] for i in indices]
        within_source_tomorrow_cluster[source] += articles

    within_source_in_cluster_indices = get_articles_in_cluster(articles_day1, dct, tfidf_model, threshold=threshold)

    for idx, indices in within_source_in_cluster_indices:
        source = articles_day1[idx].source
        articles = [articles_day1[i] for i in indices if i != idx]
        within_source_in_cluster[source] += articles

    return within_source_tomorrow_cluster, within_source_in_cluster


def get_within_source_cluster_of_articles(path, dct, tfidf_model, year, month, threshold):
    """
    Calculate different clusters of news articles for a given month and return the list of articles which are in
    cluster, which are not in cluster with any other article or articles which are not in cluster with any other
    article from the same day but in cluster with articles from next day.

    The method makes use of parallel processing and the 4 groups or clusters are calculated in parallel for each day and
    later combined to form lists for entire month.
    Parameters
    ----------
    @path: (string) path to location of articles database
    @dct: (gensim dictionary object)
    @tfidf_model: gensim tfidf model
    @year: (int)
    @month: (int)
    @threshold: (float) the cosine similarity threshold which is used to classify articles into same cluster

    Returns
    -------
    3 different list of articles, in_cluster, not_in_cluster, not_in_cluster_but_tomorrow's cluster
    """

    assert (1 > threshold > 0)
    start_date = date(year, month, 1)
    end_date = date(year, month, calendar.monthrange(year, month)[1])  # calendar.monthrange(year, month)[1]

    within_source_tomorrow_cluster = {source: [] for source in helpers.source_names}
    within_source_in_cluster = {source: [] for source in helpers.source_names}

    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    print('calculating clusters for {} {}'.format(year, calendar.month_name[month]))
    pool = mp.Pool(mp.cpu_count())  # calculate stats for date in parallel
    stats = pool.starmap(get_within_source_cluster_for_the_day, [(curr_date, path, dct, tfidf_model, threshold)
                                                                 for curr_date in date_range])
    pool.close()

    for stat in stats:
        within_source_tomorrow_cluster = helpers.combine_two_dictionary(within_source_tomorrow_cluster, stat[0])
        within_source_in_cluster = helpers.combine_two_dictionary(within_source_in_cluster, stat[1])

    return within_source_tomorrow_cluster, within_source_in_cluster


def get_cluster_for_the_day_with_tomorrows_articles(curr_date, path, dct, tfidf_model, threshold):
    """
    """
    delta = timedelta(days=1)
    next_date = curr_date + delta
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

    unclustered_articles_in_day2_cluster = [unclustered_articles[i] for i, idx in
                                            unclustered_articles_indices_in_day2_cluster]

    tomorrows_articles = []
    all_indices_of_tomorrow_articles = []

    for i, indices in unclustered_articles_indices_in_day2_cluster:
        all_indices_of_tomorrow_articles += indices

    unique_indices_of_tomorrow_articles = list(set(all_indices_of_tomorrow_articles))

    for idx in unique_indices_of_tomorrow_articles:
        tomorrows_articles += [articles_day2[idx]]

    return clustered_articles, unclustered_articles, unclustered_articles_in_day2_cluster, tomorrows_articles


def get_cluster_for_the_day(curr_date, path, dct, tfidf_model, threshold):
    """
    Parameters
    ----------
    path: (string) path to articles database
    dct: (gensim dictionary)
    tfidf_model: (gensim tfidf model)
    curr_date: (python datetime object) the date for which clusters of articles is to be calculated
    threshold: (float) the cosine similarity threshold which is used to classify articles into same cluster

    Returns
    -------
    clustered_articles: the list of articles for curr_date which are in cluster with other articles
    unclustered_articles: the list of articles for curr_date which are not in cluster with any other article
    unclustered_articles_in_day2_cluster: list of articles for curr_date which are not in cluster with any other article
    from same date but are in cluster with articles from next day
    """

    delta = timedelta(days=1)
    next_date = curr_date + delta
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

    unclustered_articles_in_day2_cluster = [unclustered_articles[i] for i, idx in
                                            unclustered_articles_indices_in_day2_cluster]

    return clustered_articles, unclustered_articles, unclustered_articles_in_day2_cluster


def get_cluster_of_articles(path, dct, tfidf_model, year, month, threshold):
    """
    Calculate different clusters of news articles for a given month and return the list of articles which are in
    cluster, which are not in cluster with any other article or articles which are not in cluster with any other
    article from the same day but in cluster with articles from next day.

    The method makes use of parallel processing and the 4 groups or clusters are calculated in parallel for each day and
    later combined to form lists for entire month.
    Parameters
    ----------
    @path: (string) path to location of articles database
    @dct: (gensim dictionary object)
    @tfidf_model: gensim tfidf model
    @year: (int)
    @month: (int)
    @threshold: (float) the cosine similarity threshold which is used to classify articles into same cluster

    Returns
    -------
    3 different list of articles, in_cluster, not_in_cluster, not_in_cluster_but_tomorrow's cluster
    """

    assert (1 > threshold > 0)
    start_date = date(year, month, 1)
    end_date = date(year, month, calendar.monthrange(year, month)[1])  # calendar.monthrange(year, month)[1]

    in_cluster_articles = []
    not_in_cluster_articles = []
    not_in_cluster_but_next_day_cluster = []

    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    print('calculating clusters for {} {}'.format(year, calendar.month_name[month]))
    pool = mp.Pool(mp.cpu_count())  # calculate stats for date in parallel
    stats = pool.starmap(get_cluster_for_the_day, [(curr_date, path, dct, tfidf_model, threshold)
                                                   for curr_date in date_range])
    pool.close()

    for stat in stats:
        in_cluster_articles += stat[0]
        not_in_cluster_articles += stat[1]
        not_in_cluster_but_next_day_cluster += stat[2]

    return in_cluster_articles, not_in_cluster_articles, not_in_cluster_but_next_day_cluster


def get_cluster_of_non_copy_articles(path, dct, tfidf_model, year, month, threshold):
    """
    Get cluster of articles which are not reported next day.
    :param path:
    :param dct:
    :param tfidf_model:
    :param year:
    :param month:
    :param threshold:
    :return:
    """

    assert (1 > threshold > 0)
    start_date = date(year, month, 1)
    end_date = date(year, month, calendar.monthrange(year, month)[1])  # calendar.monthrange(year, month)[1]

    in_cluster_articles = []
    not_in_cluster_articles = []
    tomorrow_articles = []

    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    print('calculating clusters for {} {}'.format(year, calendar.month_name[month]))
    pool = mp.Pool(mp.cpu_count())  # calculate stats for date in parallel
    stats = pool.starmap(
        get_cluster_for_the_day_with_tomorrows_articles,
        [(curr_date, path, dct, tfidf_model, threshold) for curr_date in date_range]
    )
    pool.close()

    for stat in stats:
        in_cluster_articles += stat[0]
        not_in_cluster_articles += stat[1]
        tomorrow_articles += stat[3]

    # remove tomorrow articles from in cluster and not in cluster articles
    in_cluster_articles_no_copy = [
        article for article in in_cluster_articles if article not in tomorrow_articles
    ]

    not_in_cluster_articles_no_copy = [
        article for article in not_in_cluster_articles if article not in tomorrow_articles
    ]
    return in_cluster_articles_no_copy, not_in_cluster_articles_no_copy


def get_cluster_of_articles_group_by_median_for_date(curr_date, path, dct, tfidf_model, overall_medians, source_medians,
                                                     threshold):
    """
    Parameters
    ----------
    :param threshold:
    :param tfidf_model:
    :param dct:
    :param path:
    :param curr_date:
    :param source_medians:
    :param overall_medians:

    Returns
    -------
    """

    delta = timedelta(days=1)
    next_date = curr_date + delta
    conn = db.NewsDb(path)

    print('calculating median clusters for {}'.format(curr_date))
    median_clusters = {cluster_name: [] for cluster_name in helpers.median_clusters_name}

    articles_day1 = list(conn.select_articles_by_date(curr_date))
    articles_day2 = list(conn.select_articles_by_date(next_date))

    conn.close()

    unclustered_articles_indices = get_articles_not_in_cluster(articles_day1, dct, tfidf_model, threshold=threshold)
    unclustered_articles = [articles_day1[i] for i in unclustered_articles_indices]
    unclustered_articles_indices_in_day2_cluster = get_similar_articles(unclustered_articles, articles_day2, dct,
                                                                        tfidf_model, threshold=threshold)

    clustered_articles_indices = get_articles_in_cluster(articles_day1, dct, tfidf_model, threshold=threshold)

    for i, indices in clustered_articles_indices:
        source = articles_day1[i].source
        # check for overall median clusters
        if len(indices) <= (overall_medians['in_cluster'] - 1):
            median_clusters['overall_in_cluster_below_median'].append(articles_day1[i])
        else:
            median_clusters['overall_in_cluster_above_median'].append(articles_day1[i])

        # check for source median clusters
        if len(indices) <= (source_medians['in_cluster'][source] - 1):
            median_clusters['source_in_cluster_below_median'].append(articles_day1[i])
        else:
            median_clusters['source_in_cluster_above_median'].append(articles_day1[i])

    for i, indices in unclustered_articles_indices_in_day2_cluster:
        source = articles_day1[i].source

        if len(indices) <= (overall_medians['tomorrow_cluster'] - 1):
            median_clusters['overall_in_tomorrows_cluster_below_median'].append(unclustered_articles[i])
        else:
            median_clusters['overall_in_tomorrows_cluster_above_median'].append(unclustered_articles[i])

        if len(indices) <= (source_medians['tomorrow'][source] - 1):
            median_clusters['source_in_tomorrows_cluster_below_median'].append(unclustered_articles[i])
        else:
            median_clusters['source_in_tomorrows_cluster_above_median'].append(unclustered_articles[i])

    return median_clusters


def get_cluster_of_articles_group_by_median(path, dct, tfidf_model, year, month, threshold):
    """
    Parameters
    ----------
    @path: (string) path to location of articles database
    @dct: (gensim dictionary object)
    @tfidf_model: gensim tfidf model
    @year: (int)
    @month: (int)
    @threshold: (float) the cosine similarity threshold which is used to classify articles into same cluster

    Returns
    -------
    3 different list of articles, in_cluster, not_in_cluster, not_in_cluster_but_tomorrow's cluster
    """

    assert (1 > threshold > 0)
    start_date = date(year, month, 1)
    end_date = date(year, month, calendar.monthrange(year, month)[1])  # calendar.monthrange(year, month)[1]

    median_clusters = {cluster_name: [] for cluster_name in helpers.median_clusters_name}

    overall_medians = helpers.load_json('../data/median_cluster_size_overall_{}.json'.format(year))
    source_medians = helpers.load_json('../data/median_cluster_size_by_source_{}.json'.format(year))

    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    print('calculating median clusters for {} {}'.format(year, calendar.month_name[month]))
    pool = mp.Pool(mp.cpu_count())
    stats = pool.starmap(get_cluster_of_articles_group_by_median_for_date,
                         [(curr_date, path, dct, tfidf_model, overall_medians, source_medians, threshold)
                          for curr_date in date_range])
    pool.close()

    for stat in stats:
        for cluster in helpers.median_clusters_name:
            median_clusters[cluster] += stat[cluster]

    return median_clusters


def get_clusters_and_size_for_day(curr_date, path, dct, tfidf_model, threshold=0.3):
    """
    Parameters
    ----------
    path: (string) path to articles database
    dct: (gensim dictionary)
    tfidf_model: (gensim tfidf model)
    curr_date: (python datetime object) the date for which clusters of articles is to be calculated
    threshold: (float) the cosine similarity threshold which is used to classify articles into same cluster

    Returns
    -------
    clustered_sizes:
    """

    delta = timedelta(days=1)
    next_date = curr_date + delta
    conn = db.NewsDb(path)

    print('calculating clusters for {}'.format(curr_date))

    articles_day1 = list(conn.select_articles_by_date(curr_date))
    articles_day2 = list(conn.select_articles_by_date(next_date))

    conn.close()

    cluster_sizes_for_tomorrows_cluster = {source: [] for source in helpers.source_names}
    cluster_sizes_for_in_cluster = {source: [] for source in helpers.source_names}

    unclustered_articles_indices = get_articles_not_in_cluster(articles_day1, dct, tfidf_model, threshold=threshold)
    clustered_articles_indices = get_articles_in_cluster(articles_day1, dct, tfidf_model, threshold)
    unclustered_articles = [articles_day1[i] for i in unclustered_articles_indices]
    unclustered_articles_indices_in_day2_cluster = get_similar_articles(unclustered_articles, articles_day2, dct,
                                                                        tfidf_model, threshold=threshold)

    for idx, indices in unclustered_articles_indices_in_day2_cluster:
        source = unclustered_articles[idx].source
        cluster_sizes_for_tomorrows_cluster[source] += [len(indices) + 1]

    for idx, indices in clustered_articles_indices:
        source = articles_day1[idx].source
        cluster_sizes_for_in_cluster[source] += [len(indices) + 1]

    return cluster_sizes_for_tomorrows_cluster, cluster_sizes_for_in_cluster


def get_clusters_and_median_size_for_month(path, dct, tfidf_model, year, month, agg_later=False, threshold=0.3):
    """Fetches the cluster and it's average size

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
    start_date = date(year, month, 1)
    end_date = date(year, month, calendar.monthrange(year, month)[1])  # calendar.monthrange(year, month)[1]

    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    print('calculating clusters sizes for {} {}'.format(year, calendar.month_name[month]))

    pool = mp.Pool(mp.cpu_count())  # calculate stats for date in parallel
    stats = pool.starmap(get_clusters_and_size_for_day, [(curr_date, path, dct, tfidf_model, threshold)
                                                         for curr_date in date_range])
    pool.close()

    tomorrow_clusters_stat = [stat[0] for stat in stats]
    in_cluster_stat = [stat[1] for stat in stats]

    if agg_later:
        return tomorrow_clusters_stat, in_cluster_stat

    agg_tomorrow_clusters_stat = helpers.combine_list_of_dictionary(tomorrow_clusters_stat)
    agg_in_cluster_stat = helpers.combine_list_of_dictionary(in_cluster_stat)

    rows = []
    for source, cluster_sizes in agg_tomorrow_clusters_stat.items():
        rows += [[source, len(cluster_sizes), np.median(cluster_sizes)]]

    clusters_median = pd.DataFrame(rows, columns=['Source', 'Number of Clusters', 'Size of Cluster'])
    clusters_median.to_csv(path_or_buf='../results/tomorrow_cluster_sizes_{}_{}.csv'.format(year, month))

    rows = []
    for source, cluster_sizes in agg_in_cluster_stat.items():
        rows += [[source, len(cluster_sizes), np.median(cluster_sizes)]]

    clusters_median = pd.DataFrame(rows, columns=['Date', 'Number of Clusters', 'Size of Cluster'])
    clusters_median.to_csv(path_or_buf='../results/in_cluster_sizes_{}_{}.csv'.format(year, month))

    tomorrow_clusters_stat_all_source = helpers.flatten(agg_tomorrow_clusters_stat.values())
    in_clusters_stat_all_source = helpers.flatten(agg_in_cluster_stat.values())

    rows = []
    rows += [['tomorrow_cluster', len(tomorrow_clusters_stat_all_source),
              np.median(tomorrow_clusters_stat_all_source)]]

    rows += [['in_cluster', len(in_clusters_stat_all_source), np.median(in_clusters_stat_all_source)]]
    clusters_median = pd.DataFrame(rows, columns=['cluster', 'Number of Clusters', 'Size of Cluster'])
    clusters_median.to_csv(path_or_buf='../results/overall_cluster_sizes_{}_{}.csv'.format(year, month))


def get_clusters_and_median_size_for_year(path, dct, tfidf_model, year, threshold=0.3):
    assert (1 > threshold > 0)
    stats = []
    tomorrow_clusters_stat = []
    in_cluster_stat = []

    for month in range(1, 12 + 1):
        stats += [get_clusters_and_median_size_for_month(path, dct, tfidf_model, year, month, agg_later=True,
                                                         threshold=threshold)]

    for stat in stats:
        tomorrow_clusters_stat += stat[0]
        in_cluster_stat += stat[1]

    agg_tomorrow_clusters_stat = helpers.combine_list_of_dictionary(tomorrow_clusters_stat)
    agg_in_cluster_stat = helpers.combine_list_of_dictionary(in_cluster_stat)

    rows = []
    for source, cluster_sizes in agg_tomorrow_clusters_stat.items():
        rows += [[source, len(cluster_sizes), np.median(cluster_sizes)]]

    clusters_median = pd.DataFrame(rows, columns=['Source', 'Number of Clusters', 'Size of Cluster'])
    clusters_median.to_csv(path_or_buf='../results/tomorrow_cluster_sizes_{}.csv'.format(year))

    rows = []
    for source, cluster_sizes in agg_in_cluster_stat.items():
        rows += [[source, len(cluster_sizes), np.median(cluster_sizes)]]

    clusters_median = pd.DataFrame(rows, columns=['Date', 'Number of Clusters', 'Size of Cluster'])
    clusters_median.to_csv(path_or_buf='../results/in_cluster_sizes_{}.csv'.format(year))

    tomorrow_clusters_stat_all_source = helpers.flatten(agg_tomorrow_clusters_stat.values())
    in_clusters_stat_all_source = helpers.flatten(agg_in_cluster_stat.values())

    rows = []
    rows += [['tomorrow_cluster', len(tomorrow_clusters_stat_all_source),
              np.median(tomorrow_clusters_stat_all_source)]]

    rows += [['in_cluster', len(in_clusters_stat_all_source), np.median(in_clusters_stat_all_source)]]
    clusters_median = pd.DataFrame(rows, columns=['cluster', 'Number of Clusters', 'Size of Cluster'])
    clusters_median.to_csv(path_or_buf='../results/overall_cluster_sizes_{}.csv'.format(year))


def _add_percentages_to_result(stats):
    """
    The function takes in a pandas dataframe that contains cluster statistics and adds columns with
    percent_of_unclustered_articles and percent_of_articles_in_next_day_cluster in the result.

    Parameters
    ----------
    @stats: pandas DataFrame, containing the cluster statistics

    Returns
    -------
    Pandas DataFrame with added columns to represent percentages in statistics
    """

    stats['percent_of_unclustered_articles'] = (stats['unclustered_articles'] /
                                                stats['total_articles']).fillna(0).round(2)
    stats['percent_of_articles_in_next_day_cluster'] = (
            stats['unclustered_articles_in_next_day_cluster'] /
            stats['unclustered_articles']).fillna(0).round(2)


def aggregate_by_year(path, dct, tfidf_model, year, threshold=0.3):
    """aggregate result across year by source and month

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
        stats += [aggregate_by_month(path, dct, tfidf_model, year, month, agg_later=True, threshold=threshold)]

    stats = pd.concat(stats)
    stats_by_source = stats.drop(columns=['month']).groupby(['source']).mean().astype(int)  # group by source
    stats_by_month = stats.drop(columns=['source']).groupby(['month']).sum()  # group by date

    # add percentages to stats
    _add_percentages_to_result(stats_by_month)
    _add_percentages_to_result(stats_by_source)

    return stats_by_source, stats_by_month


def aggregate_by_month(path, dct, tfidf_model, year, month, agg_later=False, threshold=0.3):
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
    stats = list(zip(*stats))
    stats = pd.concat(stats[0])

    # average the stats for month, to be used by function aggregate by year, which reports results averaged by month
    if agg_later:
        stats = stats.drop(columns=['date'])
        stats['month'] = month_name[month]
        return stats

    stats_by_source = stats.drop(columns=['date']).groupby(['source']).mean().astype(int)  # group by source
    stats_by_date = stats.drop(columns=['source']).groupby(['date']).sum()  # group by date

    # add percentages to stats
    _add_percentages_to_result(stats_by_date)
    _add_percentages_to_result(stats_by_source)

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


def get_stats_for_date(path, dct, tfidf_model, curr_date, threshold=0.3):
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
    conn = db.NewsDb(path)

    articles_day1 = list(conn.select_articles_by_date(curr_date))
    articles_day2 = list(conn.select_articles_by_date(next_date))
    count_grouped_by_source = conn.get_count_of_articles_for_date_by_source(curr_date)

    conn.close()  # close database connection

    assert (articles_day1 is not None and articles_day2 is not None)

    results = initialize_results(count_grouped_by_source)
    unclustered_articles_indices = get_articles_not_in_cluster(articles_day1, dct, tfidf_model, threshold=threshold)
    unclustered_articles = [articles_day1[i] for i in unclustered_articles_indices]
    unclustered_articles_indices_in_day2_cluster = get_similar_articles(unclustered_articles, articles_day2, dct,
                                                                        tfidf_model, threshold=threshold)
    unclustered_articles_in_day2_cluster = [unclustered_articles[i] for i, idx in
                                            unclustered_articles_indices_in_day2_cluster]

    assert (unclustered_articles is not None and unclustered_articles_in_day2_cluster is not None)

    # update the count of unclustered articles for current date
    for itr in unclustered_articles:
        source = itr.source
        results[source][1] += 1

    # update count of articles that form cluster in next day
    for it in unclustered_articles_in_day2_cluster:
        source = it.source
        results[source][2] += 1

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

    dct = corpora.Dictionary.load(args.dictionary)
    tfidf_model = models.TfidfModel.load(args.tfidf_model)
    year = args.year
    month = args.month
    threshold = args.threshold
    db_path = args.db_path
    cluster_median = args.cluster_median

    if cluster_median:
        if month is None:
            get_clusters_and_median_size_for_year(db_path, dct, tfidf_model, year, threshold=0.3)
        else:
            get_clusters_and_median_size_for_month(db_path, dct, tfidf_model, year, month, agg_later=False,
                                                   threshold=0.3)
    else:
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
