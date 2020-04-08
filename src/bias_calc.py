import argparse
import pandas as pd
from gensim import corpora, models
import cluster_analysis
import parmap
from tqdm import tqdm
import multiprocessing as mp
import copy
import numpy as np
import db
import helpers
import bigrams

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

parser.add_argument('-pb', '--parliament-bigrams',
                    type=str,
                    required=True,
                    help='the path to top n bigrams from parliament speeches')

parser.add_argument('-g', '--group-by',
                    type=str,
                    choices=['source_id', 'source_name'],
                    default='source_name',
                    help='whether bias is categorized by source id or source name. Multiple source id can have same'
                         'name because online and print version have different id')

parser.add_argument('-bt', '--bias-type',
                    type=str,
                    choices=['general', 'within_source', 'median_groups', 'standardization_parameters'],
                    default='general',
                    help='')


def get_bigrams_for_median_clusters(db_path, dct, tfidf_model, year, month, group_by, threshold=0.3):
    """
    A wrapper function to return the bigrams present in the news articles for the given year and month for different
    median clusters.
    Parameters
    ----------
    db_path: (string) path to the articles database
    dct: (gensim dictionary object)
    tfidf_model: (gensim tfidf model)
    year: (int)
    month: (int)
    group_by: (string)
    threshold: (float) the threshold for cosine similarity which is used to determine if two articles are in same
    cluster or not.

    Returns
    -------
    bigrams_above_median:
    bigrams_below_median:
    bigrams_above_median_in_cluster:
    bigrams_below_median_in_cluster:
    """

    median_clusters = cluster_analysis.get_cluster_of_articles_group_by_median(db_path, dct, tfidf_model, year, month,
                                                                               threshold)
    bigrams_for_median_clusters = {cluster_name: [] for cluster_name in helpers.median_clusters_name}

    for cluster_type, cluster in median_clusters.items():
        print(type(cluster))
        bigrams_for_median_clusters[cluster_type] = bigrams.get_bigrams_in_articles(cluster, group_by)

    return bigrams_for_median_clusters


def get_bigrams_for_within_source_clusters(db_path, dct, tfidf_model, year, month, group_by, threshold=0.3):
    """
    A wrapper function to return the bigrams present in the news articles for the given year and month for different
    clusters.
    Parameters
    ----------
    db_path: (string) path to the articles database
    dct: (gensim dictionary object)
    tfidf_model: (gensim tfidf model)
    year: (int)
    month: (int)
    group_by: (string)
    threshold: (float) the threshold for cosine similarity which is used to determine if two articles are in same
    cluster or not.

    Returns
    -------
    bigrams_within_source_tomorrow_cluster:
    bigrams_within_source_in_cluster:
    """
    within_source_tomorrow_cluster, within_source_in_cluster = \
        cluster_analysis.get_within_source_cluster_of_articles(db_path, dct, tfidf_model, year, month, threshold)

    bigrams_within_source_tomorrow_cluster = {}
    bigrams_within_source_in_cluster = {}

    print('calculating bigrams for within source tomorrow articles')
    for source, articles in tqdm(within_source_tomorrow_cluster.items()):
        bigrams_within_source_tomorrow_cluster[source] = bigrams.get_bigrams_in_articles(articles, group_by,
                                                                                         pbar=False)

    print('calculating bigrams for within source in cluster articles')
    for source, articles in tqdm(within_source_in_cluster.items()):
        bigrams_within_source_in_cluster[source] = bigrams.get_bigrams_in_articles(articles, group_by, pbar=False)

    return bigrams_within_source_tomorrow_cluster, bigrams_within_source_in_cluster


def get_bigrams_by_clusters(db_path, dct, tfidf_model, year, month, group_by, threshold=0.3):
    """
    A wrapper function to return the bigrams present in the news articles for the given year and month for different
    clusters.
    Parameters
    ----------
    db_path: (string) path to the articles database
    dct: (gensim dictionary object)
    tfidf_model: (gensim tfidf model)
    year: (int)
    month: (int)
    group_by: (string)
    threshold: (float) the threshold for cosine similarity which is used to determine if two articles are in same
    cluster or not.

    Returns
    -------
    bigrams_in_cluster: (dictionary) key as source and value as all the bigrams present in the news for the particular
                        source and articles which are in cluster for given month and year
    bigrams_not_in_cluster: (dictionary)
    bigrams_in_cluster_tomorrow:
    bigrams_all_articles:
    """

    conn = db.NewsDb(db_path)
    n_articles = conn.get_count_of_articles_for_year_and_month(year, month)
    conn.close()

    in_cluster, not_in_cluster, in_cluster_tomorrow = cluster_analysis.get_cluster_of_articles(db_path, dct,
                                                                                               tfidf_model,
                                                                                               year, month,
                                                                                               threshold)

    all_articles = in_cluster + not_in_cluster

    assert (n_articles == (len(in_cluster) + len(not_in_cluster)))
    assert (len(all_articles) == (len(in_cluster) + len(not_in_cluster)))

    print('calculating bigrams for in cluster articles')
    bigrams_in_cluster = bigrams.get_bigrams_in_articles(in_cluster, group_by)
    print('calculating bigrams for not in cluster articles')
    bigrams_not_in_cluster = bigrams.get_bigrams_in_articles(not_in_cluster, group_by)
    print('calculating bigrams for in tomorrow cluster articles')
    bigrams_in_cluster_tomorrow = bigrams.get_bigrams_in_articles(in_cluster_tomorrow, group_by)
    print('calculating bigrams for all articles')
    bigrams_all_articles = bigrams.get_bigrams_in_articles(all_articles, group_by)

    return bigrams_in_cluster, bigrams_not_in_cluster, bigrams_in_cluster_tomorrow, bigrams_all_articles


def standardization_helper(shares_to_standardize, overall_params, stacked_params):
    """
    Take in list of bigrams shares and standardize it using three methods.

    :param shares_to_standardize: list of shares to standardize
    :param overall_params:
    :param stacked_params:
    :return:
    """
    result = {'specific': [], 'stacked': [], 'overall': []}

    assert (overall_params is not None)
    assert (stacked_params is not None)

    for shares in shares_to_standardize:
        result['specific'].append(bigrams.standardize_bigrams_count(shares.copy()))

    for shares in shares_to_standardize:
        result['stacked'].append(bigrams.standardize_with_mean_and_std(shares.copy(), stacked_params))

    for shares in shares_to_standardize:
        result['overall'].append(bigrams.standardize_with_mean_and_std(shares.copy(), overall_params))

    return result


def standardization_helper_within_source(top_bigram_shares, overall_params, stacked_params):
    shares_specific = copy.deepcopy(top_bigram_shares)
    shares_overall = copy.deepcopy(top_bigram_shares)
    shares_stacked = copy.deepcopy(top_bigram_shares)

    bigrams.standardize_bigrams_count_group_by_source(shares_specific)
    bigrams.standardize_with_mean_and_std_group_by_source(shares_overall, overall_params)
    bigrams.standardize_with_mean_and_std_group_by_source(shares_stacked, stacked_params)

    top_bigram_shares_standardized = {
        'specific': shares_specific,
        'overall': shares_overall,
        'stacked': shares_stacked,
    }
    return top_bigram_shares_standardized


def get_shares_of_top_1000_bigrams_for_source(source, bigram_freq, top1000_bigram):
    """
    Helper function for get_shares_of_top1000_bigrams which calculate shares of bigram in a given source

    Parameters
    ----------
    source
    top1000_bigram
    bigram_freq

    Returns
    -------

    """
    top1000_bigram_freq = [0] * 1000

    for i in range(1000):
        if top1000_bigram[i] in bigram_freq:
            top1000_bigram_freq[i] = bigram_freq[top1000_bigram[i]]

    row = [source] + top1000_bigram_freq

    return row


def get_shares_of_top1000_bigrams(top1000_bigram, bigrams_by_source, pbar=True):
    """
    Calculates the shares of top 1000 bigrams occurring in bigrams grouped by news source
    Parameters
    ----------
    @top1000_bigram: (list) of 1000 bigrams from MP speeches
    @bigrams: (dictionary) of bigrams calculated from news source with key as source

    Returns
    -------
    a pandas DataFrame with count of each top 1000 bigrams in different news source
    """

    assert (len(top1000_bigram) == 1000)

    rows = parmap.starmap(get_shares_of_top_1000_bigrams_for_source, bigrams_by_source.items(), top1000_bigram,
                          pm_pbar=pbar)

    columns = ['source'] + top1000_bigram

    return pd.DataFrame(rows, columns=columns).sort_values(by=['source']).reset_index(drop=True)


def get_shares_of_top1000_bigrams_grouped_by_source(top1000_bigram, bigrams_by_source):
    """

    Parameters
    ----------
    bigrams_by_source
    top1000_bigram

    Returns
    -------

    """
    assert (len(top1000_bigram) == 1000)
    shares_by_source = {key: None for key in bigrams_by_source.keys()}

    for source, bigrams_by_source in tqdm(bigrams_by_source.items()):
        shares_by_source[source] = get_shares_of_top1000_bigrams(top1000_bigram, bigrams_by_source, pbar=False)

    return shares_by_source


def calculate_bias(top1000_bigrams_freq_by_source, top1000bigrams):
    """

    Parameters
    ----------
    @top1000_bigrams_freq_by_source: (dictionary) frequency of top 1000 bigrams grouped by source
    @top1000bigrams: (DataFrame) with top1000 bigrams and alpha and beta bias coefficients

    Returns
    -------
    bias of each source
    """

    top_bigrams = top1000_bigrams_freq_by_source.columns.tolist()[1:]

    assert (len(top_bigrams) == 1000)

    bias_by_source = {}

    for index, row in top1000_bigrams_freq_by_source.iterrows():
        num = 0
        den = 0
        # print('count of source {} is {} '.format(row['source'], bigrams_count_by_source[row['source']]))
        for i in range(1000):
            alpha = top1000bigrams[top1000bigrams['bigram'] == top_bigrams[i]].iloc[0]['alpha']
            beta = top1000bigrams[top1000bigrams['bigram'] == top_bigrams[i]].iloc[0]['beta']
            bigram_share = row[top_bigrams[i]]
            assert (isinstance(alpha, float))
            assert (isinstance(beta, float))
            # assert (isinstance(bigram_share, float))

            # print('alpha: {}, beta: {}'.format(alpha, beta))
            num += beta * (bigram_share - alpha)
            den += beta * beta
        # print('numerator: {}, denominator: {}'.format(num, den))
        bias = num / den
        bias_by_source[row['source']] = bias

    return bias_by_source


def calculate_bias_group_by_source_helper(source, aggregate_shares, top1000_bigram):
    bias = calculate_bias(aggregate_shares[source], top1000_bigram)
    row = [source]
    for source_name in helpers.source_names:
        row += [bias[source_name]]
    return row


def calculate_bias_group_by_source(aggregate_shares, top1000_bigram):
    columns = ['source'] + helpers.source_names
    sources = aggregate_shares.keys()

    rows = parmap.map(calculate_bias_group_by_source_helper, sources, aggregate_shares, top1000_bigram, pm_pbar=True)

    return pd.DataFrame(rows, columns=columns).sort_values(by=['source']).reset_index(drop=True)


def _combine_bias_result_for_all_cluster(columns, *args):
    """
    Parameters
    ----------
    @all_articles: (list)
    @in_cluster: (list)
    @not_in_cluster: (list)
    @in_tomorrow_cluster: (list)

    Returns
    -------
    a pandas dataframe with bias for different cluster combined
    """

    assert ('source' in columns)
    assert (len(columns) == (len(args) + 1))
    rows = []

    for source in helpers.source_names:
        row = [source]
        for cluster in args:
            bias_for_cluster = cluster[source] if source in cluster else 0
            row += [bias_for_cluster]
        rows += [row]

    return pd.DataFrame(rows, columns=columns).sort_values(by=['source']).reset_index(drop=True)


def bias_averaged_over_month_for_median_clusters(db_path, dct, tfidf_model, top1000_bigram, year, month, group_by,
                                                 threshold=0.3):
    """

    Parameters
    ----------
    db_path
    dct
    tfidf_model
    top1000_bigram
    year
    month
    group_by
    threshold

    Returns
    -------

    """
    bigrams_for_clusters = get_bigrams_for_median_clusters(db_path, dct, tfidf_model, year, month, group_by,
                                                           threshold=threshold)

    top_bigrams = top1000_bigram['bigram'].tolist()

    top_shares = {cluster_name: [] for cluster_name in helpers.median_clusters_name}

    print('converting bigrams list to fractional count')
    for key, val in bigrams_for_clusters.items():
        bigrams.convert_bigrams_to_shares(val)
        print('get top bigrams share for {}'.format(key))
        top_shares[key] = get_shares_of_top1000_bigrams(top_bigrams, val)

    all_mean_and_std = helpers.load_json('../data/all_mean_and_std_{}_{}.json'.format(year, month))
    stacked_mean_and_std = helpers.load_json('../data/stacked_mean_and_std_{}_{}.json'.format(year, month))

    shares = [top_shares[key] for key in helpers.median_clusters_name]

    standardized_shares = standardization_helper(shares, all_mean_and_std, stacked_mean_and_std)
    flatten_std_shares = (standardized_shares['specific'] + standardized_shares['overall'] +
                          standardized_shares['stacked'])

    assert (len(flatten_std_shares) == 3 * len(shares))

    print('calculating bias of news source by cluster groups')
    bias_result = parmap.map(calculate_bias, flatten_std_shares, top1000_bigram, pm_pbar=True)

    assert (len(bias_result) == 24)
    overall_bias_specific_std = bias_result[0:4]
    source_bias_specific_std = bias_result[4:8]
    overall_bias_overall_std = bias_result[8:12]
    source_bias_overall_std = bias_result[12:16]
    overall_bias_stacked_std = bias_result[16:20]
    source_bias_stacked_std = bias_result[20:24]

    columns = ['source', 'bias_for_above_median_tomorrow', 'bias_for_below_median_tomorrow',
               'bias_for_above_median_in_cluster', 'bias_for_below_in_cluster']
    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *overall_bias_specific_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_overall_median_{}_{}_std=specific.csv'.format(year, month))

    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *overall_bias_overall_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_overall_median_{}_{}_std=all.csv'.format(year, month))

    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *overall_bias_stacked_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_overall_median_{}_{}_std=stacked.csv'.format(year, month))

    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *source_bias_specific_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_source_median_{}_{}_std=specific.csv'.format(year, month))

    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *source_bias_overall_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_source_median_{}_{}_std=all.csv'.format(year, month))

    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *source_bias_stacked_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_source_median_{}_{}_std=stacked.csv'.format(year, month))


def bias_averaged_over_month_for_within_source_clusters(db_path, dct, tfidf_model, top1000_bigram, year, month,
                                                        group_by, threshold=0.3):
    bigrams_within_source_tomorrow, bigrams_within_source_in_cluster = get_bigrams_for_within_source_clusters(
        db_path, dct, tfidf_model, year, month, group_by, threshold)

    bigrams.convert_bigrams_to_shares_grouped_by_source(bigrams_within_source_tomorrow)
    bigrams.convert_bigrams_to_shares_grouped_by_source(bigrams_within_source_in_cluster)

    top_bigrams = top1000_bigram['bigram'].tolist()

    print('get top bigrams for within source tomorrow cluster')
    top_bigrams_freq_within_source_tomorrow = get_shares_of_top1000_bigrams_grouped_by_source(
        top_bigrams, bigrams_within_source_tomorrow)
    print('get top bigrams for within source in cluster')
    top_bigrams_freq_within_source_in_cluster = get_shares_of_top1000_bigrams_grouped_by_source(
        top_bigrams, bigrams_within_source_in_cluster)

    all_mean_and_std = helpers.load_json('../data/all_mean_and_std_{}_{}.json'.format(year, month))
    stacked_mean_and_std = helpers.load_json('../data/stacked_mean_and_std_{}_{}.json'.format(year, month))

    within_source_tomorrow_std_shares = standardization_helper_within_source(top_bigrams_freq_within_source_tomorrow,
                                                                             all_mean_and_std, stacked_mean_and_std)
    within_source_in_cluster_std_shares = standardization_helper_within_source(
        top_bigrams_freq_within_source_in_cluster,
        all_mean_and_std, stacked_mean_and_std)

    print('calculate bias for within source tomorrow cluster specific std')
    bias_within_source = calculate_bias_group_by_source(within_source_tomorrow_std_shares['specific'], top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_tomorrow_{}_{}_std=specific.csv'.format(year,
                                                                                                                 month))

    print('calculate bias for within source tomorrow cluster overall std')
    bias_within_source = calculate_bias_group_by_source(within_source_tomorrow_std_shares['overall'], top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_tomorrow_{}_{}_std=overall.csv'.format(year,
                                                                                                                month))

    print('calculate bias for within source tomorrow cluster stacked std')
    bias_within_source = calculate_bias_group_by_source(within_source_tomorrow_std_shares['stacked'], top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_tomorrow_{}_{}_std=stacked.csv'.format(year,
                                                                                                                month))

    print('calculate bias for within source in cluster specific std')
    bias_within_source = calculate_bias_group_by_source(within_source_in_cluster_std_shares['specific'], top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_in_cluster_{}_{}_std=stacked.csv'.format(year,
                                                                                                                  month))

    print('calculate bias for within source in cluster overall std')
    bias_within_source = calculate_bias_group_by_source(within_source_in_cluster_std_shares['overall'], top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_in_cluster_{}_{}_std=stacked.csv'.format(year,
                                                                                                                  month))

    print('calculate bias for within source in cluster stacked std')
    bias_within_source = calculate_bias_group_by_source(within_source_in_cluster_std_shares['stacked'], top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_in_cluster_{}_{}_std=stacked.csv'.format(year,
                                                                                                                  month))


def bias_averaged_over_month(db_path, dct, tfidf_model, top1000_bigram, year, month, group_by, threshold=0.3):
    """
    Parameters
    ----------
    @db_path: (string) path to articles database
    @dct: (gensim dictionary object)
    @tfidf_model: (gensim tfidf object)
    @top1000_bigram: top 1000 bigrams in MP speeches with alpha and beta bias coefficient
    @month: (int)
    @year: (int)
    @std_type:
    @threshold: (float)

    Returns
    -------
    """
    top_bigrams = top1000_bigram['bigram'].tolist()

    bigrams_in_cluster, bigrams_not_in_cluster, bigrams_in_cluster_tomorrow, bigrams_all_articles = \
        get_bigrams_by_clusters(db_path, dct, tfidf_model, year, month, group_by, threshold)

    print('converting bigrams list to fractional count')
    total_bigrams_all_articles = bigrams.convert_bigrams_to_shares(bigrams_all_articles)
    total_bigrams_in_cluster = bigrams.convert_bigrams_to_shares(bigrams_in_cluster)
    total_bigrams_not_in_cluster = bigrams.convert_bigrams_to_shares(bigrams_not_in_cluster)
    total_bigrams_in_cluster_tomorrow = bigrams.convert_bigrams_to_shares(bigrams_in_cluster_tomorrow)

    print('get top bigrams share for all articles')
    top_bigrams_freq_all_articles = get_shares_of_top1000_bigrams(top_bigrams, bigrams_all_articles)
    print('get top bigrams share for in cluster')
    top_bigrams_freq_in_cluster = get_shares_of_top1000_bigrams(top_bigrams, bigrams_in_cluster)
    print('get top bigrams for not in cluster')
    top_bigrams_freq_not_in_cluster = get_shares_of_top1000_bigrams(top_bigrams, bigrams_not_in_cluster)
    print('get top bigrams for in cluster tomorrow')
    top_bigrams_freq_in_cluster_tomorrow = get_shares_of_top1000_bigrams(top_bigrams, bigrams_in_cluster_tomorrow)

    del bigrams_in_cluster, bigrams_in_cluster_tomorrow, bigrams_not_in_cluster, bigrams_all_articles

    top_bigrams_freq_all_articles.to_csv(path_or_buf='../results/bigrams_share_all_articles_{}_{}.csv'.format(
        year, month))
    top_bigrams_freq_in_cluster.to_csv(path_or_buf='../results/bigrams_share_in_cluster_{}_{}.csv'.format(
        year, month))
    top_bigrams_freq_not_in_cluster.to_csv(path_or_buf='../results/bigrams_share_not_in_cluster_{}_{}.csv'.format(
        year, month))
    top_bigrams_freq_in_cluster_tomorrow.to_csv(
        path_or_buf='../results/bigrams_share_in_cluster_tomorrow_{}_{}.csv'.format(year, month))

    helpers.save_json(total_bigrams_all_articles, '../results/total_bigrams_all_articles_{}_{}.json'.format(year,
                                                                                                            month))
    helpers.save_json(total_bigrams_in_cluster, '../results/total_bigrams_in_cluster_{}_{}.json'.format(year,
                                                                                                        month))
    helpers.save_json(total_bigrams_not_in_cluster, '../results/total_bigrams_not_in_cluster_{}_{}.json'.format(
        year, month))
    helpers.save_json(total_bigrams_in_cluster_tomorrow, '../results/total_bigrams_in_cluster_tomorrow_{}_{}.json'.
                      format(year, month))

    all_mean_and_std = bigrams.get_mean_and_deviation(top_bigrams_freq_all_articles)
    helpers.save_json(all_mean_and_std, '../data/all_mean_and_std_{}_{}.json'.format(year, month))
    mean_and_std = bigrams.get_stacked_mean_and_deviation(top_bigrams_freq_in_cluster, top_bigrams_freq_not_in_cluster)
    helpers.save_json(mean_and_std, '../data/stacked_mean_and_std_{}_{}.json'.format(year, month))

    shares = [top_bigrams_freq_all_articles, top_bigrams_freq_in_cluster, top_bigrams_freq_not_in_cluster,
              top_bigrams_freq_in_cluster_tomorrow]
    standardized_shares = standardization_helper(shares, all_mean_and_std, mean_and_std)
    flatten_std_shares = (standardized_shares['specific'] + standardized_shares['overall'] +
                          standardized_shares['stacked'])

    assert (len(flatten_std_shares) == 12)

    print('calculating bias of news source by cluster groups')
    bias_result = parmap.map(calculate_bias, flatten_std_shares, top1000_bigram, pm_pbar=True)

    assert (len(bias_result) == 12)
    bias_specific_std = bias_result[0:4]
    bias_overall_std = bias_result[4:8]
    bias_stacked_std = bias_result[8:12]

    columns = ['source', 'bias_all_articles', 'bias_in_cluster', 'bias_not_in_cluster', 'bias_in_cluster_tomorrow']
    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *bias_specific_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_{}_{}_std=specific.csv'.format(year, month))

    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *bias_overall_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_{}_{}_std=overall.csv'.format(year, month))

    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *bias_stacked_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_{}_{}_std=stacked.csv'.format(year, month))


def aggregate_bigrams_month_count(total_bigrams_for_month):
    """

    Parameters
    ----------
    total_bigrams_for_month: (list of dictionaries) each dictionary in the list corresponds to a month and stores keys
                             as the source and value as the count of bigrams for the source in particular month.

    Returns
    -------
    total_bigrams_count: (dictionary) with key as source and values as total count of bigrams for the source
                         across all month
    """

    total_bigrams_count = {}
    n_groups = len(total_bigrams_for_month)
    source_across_month = set()

    assert (len(total_bigrams_for_month) != 0)

    for i in range(n_groups):
        source_across_month.update(list(total_bigrams_for_month[i].keys()))

    source_across_month = list(source_across_month)

    for source in source_across_month:
        total_bigrams_count[source] = 0

        for i in range(n_groups):
            if source in total_bigrams_for_month[i]:
                total_bigrams_count[source] += total_bigrams_for_month[i][source]

    return total_bigrams_count


def aggregate_bigrams_month_count_grouped_by_source(total_bigrams_for_month):
    """

    Parameters
    ----------
    total_bigrams_for_month: (list of dictionaries) each dictionary in the list corresponds to a month and stores keys
                             as the source and value as the count of bigrams for the source in particular month.

    Returns
    -------
    total_bigrams_count: (dictionary) with key as source and values as total count of bigrams for the source
                         across all month
    """

    total_bigrams_count = {}
    assert (len(total_bigrams_for_month) == 12)

    for source in helpers.source_names:
        bigrams_count_for_source = []
        for i in range(12):
            bigrams_count_for_source.append(total_bigrams_for_month[i][source])

        total_bigrams_count[source] = aggregate_bigrams_month_count(bigrams_count_for_source)

    return total_bigrams_count


def aggregate_bigrams_month_share_for_source(source, top_bigrams_month_share, total_bigrams_month,
                                             aggregate_source_count):
    n_groups = len(top_bigrams_month_share)
    assert (n_groups == len(total_bigrams_month))

    weighted_shares = np.array([0] * 1000, dtype=float)
    if aggregate_source_count[source] == 0:
        row = [source] + list(weighted_shares)
        return row

    for i in range(n_groups):
        top_bigrams_shares_for_month = top_bigrams_month_share[i]
        total_bigrams_for_month = total_bigrams_month[i]
        # print(top_bigrams_shares_for_month['source'])
        if source not in top_bigrams_shares_for_month['source'].tolist():
            continue

        shares = np.array(top_bigrams_shares_for_month[top_bigrams_shares_for_month['source'] ==
                                                       source].iloc[0][1:].tolist())
        weight = total_bigrams_for_month[source]
        # print('weight: ', weight)
        # print('shares: ', shares)
        weighted_shares += shares * weight
    weighted_shares = weighted_shares / aggregate_source_count[source]

    row = [source] + weighted_shares.tolist()

    return row


def aggregate_bigrams_month_share(top_bigrams_month_share, total_bigrams_month, aggregate_source_count, top_bigrams,
                                  pbar=True):
    """

    Parameters
    ----------
    pbar
    top_bigrams_month_share: (list)
    total_bigrams_month: (list)
    aggregate_source_count: (dictionary)
    top_bigrams: (list)

    Returns
    -------
    a pandas dataframe with aggregate share of top bigrams across source
    """

    assert (len(top_bigrams_month_share) == len(total_bigrams_month))
    assert (len(top_bigrams) == 1000)

    sources = list(aggregate_source_count.keys())

    rows = []
    columns = ['source'] + top_bigrams

    rows = parmap.map(aggregate_bigrams_month_share_for_source, sources, top_bigrams_month_share, total_bigrams_month,
                      aggregate_source_count, pm_pbar=pbar)

    return pd.DataFrame(rows, columns=columns).sort_values(by=['source']).reset_index(drop=True)


def aggregate_bigrams_month_share_group_by_source(top_bigrams_month_share, total_bigrams_month, aggregate_source_count,
                                                  top_bigrams):
    aggregate_bigrams_share = {}
    n_groups = len(top_bigrams_month_share)
    assert (n_groups == len(total_bigrams_month))

    for source in tqdm(helpers.source_names):
        bigrams_month_share = [top_bigrams_month_share[i][source] for i in range(n_groups)]
        bigrams_month_count = [total_bigrams_month[i][source] for i in range(n_groups)]
        aggregate_bigrams_share[source] = aggregate_bigrams_month_share(bigrams_month_share, bigrams_month_count,
                                                                        aggregate_source_count[source], top_bigrams,
                                                                        pbar=False)

    return aggregate_bigrams_share


def bias_averaged_over_year_for_median_clusters(db_path, dct, tfidf_model, top1000_bigram, year, group_by,
                                                threshold=0.3):
    assert (1 > threshold > 0)
    top_bigrams = top1000_bigram['bigram'].tolist()

    top_bigrams_share_by_month = {cluster_name: [] for cluster_name in helpers.median_clusters_name}
    total_bigrams_by_month = {cluster_name: [] for cluster_name in helpers.median_clusters_name}
    aggregate_source_count = {cluster_name: None for cluster_name in helpers.median_clusters_name}
    aggregate_share = {cluster_name: None for cluster_name in helpers.median_clusters_name}

    # first get top bigrams shares for all months of the year and also the total count of bigrams for every source
    for month in range(1, 12 + 1):
        bigrams_for_month = get_bigrams_for_median_clusters(db_path, dct, tfidf_model, year, month, group_by,
                                                            threshold=threshold)

        print('converting bigrams list to fractional count')
        for cluster_name, cluster in bigrams_for_month.items():
            total_bigrams_by_month[cluster_name].append(bigrams.convert_bigrams_to_shares(cluster))
            print('get top bigrams share for {}'.format(cluster_name))
            top_bigrams_share_by_month[cluster_name].append(get_shares_of_top1000_bigrams(top_bigrams, cluster))

    # get aggregate bigram count across month, i.e count of bigrams in an year grouped by month
    for cluster_name in helpers.median_clusters_name:
        aggregate_source_count[cluster_name] = aggregate_bigrams_month_count(total_bigrams_by_month[cluster_name])

    # get share of top bigrams for a year by aggregating the share for each month
    for cluster_name in helpers.median_clusters_name:
        print('aggregating bigram share for {}'.format(cluster_name))
        aggregate_share[cluster_name] = aggregate_bigrams_month_share(top_bigrams_share_by_month[cluster_name],
                                                                      total_bigrams_by_month[cluster_name],
                                                                      aggregate_source_count[cluster_name],
                                                                      top_bigrams)

    all_mean_and_std = helpers.load_json('../data/all_mean_and_std_{}.json'.format(year))
    stacked_mean_and_std = helpers.load_json('../data/stacked_mean_and_std_{}.json'.format(year))

    shares = [aggregate_share[key] for key in helpers.median_clusters_name]

    standardized_shares = standardization_helper(shares, all_mean_and_std, stacked_mean_and_std)
    flatten_std_shares = (standardized_shares['specific'] + standardized_shares['overall'] +
                          standardized_shares['stacked'])

    assert (len(flatten_std_shares) == 3 * len(shares))

    print('calculating bias of news source by cluster groups')
    bias_result = parmap.map(calculate_bias, flatten_std_shares, top1000_bigram, pm_pbar=True)

    assert (len(bias_result) == 24)
    overall_bias_specific_std = bias_result[0:4]
    overall_bias_overall_std = bias_result[4:8]
    overall_bias_stacked_std = bias_result[8:12]
    source_bias_specific_std = bias_result[12:16]
    source_bias_overall_std = bias_result[16:20]
    source_bias_stacked_std = bias_result[20:24]

    columns = ['source', 'bias_for_above_median_tomorrow', 'bias_for_below_median_tomorrow',
               'bias_for_above_median_in_cluster', 'bias_for_below_in_cluster']
    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *overall_bias_specific_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_overall_median_{}_std=specific.csv'.format(year))

    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *overall_bias_overall_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_overall_median_{}_std=all.csv'.format(year))

    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *overall_bias_stacked_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_overall_median_{}_std=stacked.csv'.format(year))

    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *source_bias_specific_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_source_median_{}_std=specific.csv'.format(year))

    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *source_bias_overall_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_source_median_{}_std=all.csv'.format(year))

    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *source_bias_stacked_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_source_median_{}_std=stacked.csv'.format(year))


def bias_averaged_over_year_for_within_source_clusters(db_path, dct, tfidf_model, top1000_bigram, year, group_by,
                                                       threshold=0.3):
    assert (1 > threshold > 0)
    top_bigrams = top1000_bigram['bigram'].tolist()

    top_bigrams_share_by_month_within_source_tomorrow = []
    total_bigrams_by_month_within_source_tomorrow = []
    top_bigrams_share_by_month_within_source_in_cluster = []
    total_bigrams_by_month_within_source_in_cluster = []

    for month in range(1, 12 + 1):
        bigrams_within_source_tomorrow, bigrams_within_source_in_cluster = \
            get_bigrams_for_within_source_clusters(db_path, dct, tfidf_model, year, month, group_by, threshold)

        total_bigrams_within_source_tomorrow = bigrams.convert_bigrams_to_shares_grouped_by_source(
            bigrams_within_source_tomorrow)
        total_bigrams_within_source_in_cluster = bigrams.convert_bigrams_to_shares_grouped_by_source(
            bigrams_within_source_in_cluster)

        print('get shares of top bigrams for within source cluster tomorrow')
        top_bigrams_freq_within_source_tomorrow = get_shares_of_top1000_bigrams_grouped_by_source(
            top_bigrams, bigrams_within_source_tomorrow)
        print('get shares of top bigrams for within source in cluster')
        top_bigrams_freq_within_source_in_cluster = get_shares_of_top1000_bigrams_grouped_by_source(
            top_bigrams, bigrams_within_source_in_cluster)

        top_bigrams_share_by_month_within_source_tomorrow.append(top_bigrams_freq_within_source_tomorrow)
        total_bigrams_by_month_within_source_tomorrow.append(total_bigrams_within_source_tomorrow)

        top_bigrams_share_by_month_within_source_in_cluster.append(top_bigrams_freq_within_source_in_cluster)
        total_bigrams_by_month_within_source_in_cluster.append(total_bigrams_within_source_in_cluster)

    aggregate_source_count_within_source_tomorrow = aggregate_bigrams_month_count_grouped_by_source(
        total_bigrams_by_month_within_source_tomorrow)
    aggregate_source_count_within_source_in_cluster = aggregate_bigrams_month_count_grouped_by_source(
        total_bigrams_by_month_within_source_in_cluster)

    print('aggregating bigram share for within source tomorrow articles')
    aggregate_share_within_source_tomorrow = aggregate_bigrams_month_share_group_by_source(
        top_bigrams_share_by_month_within_source_tomorrow,
        total_bigrams_by_month_within_source_tomorrow,
        aggregate_source_count_within_source_tomorrow,
        top_bigrams)

    print('aggregating bigram share for within source in cluster articles')
    aggregate_share_within_source_in_cluster = aggregate_bigrams_month_share_group_by_source(
        top_bigrams_share_by_month_within_source_in_cluster,
        total_bigrams_by_month_within_source_in_cluster,
        aggregate_source_count_within_source_in_cluster,
        top_bigrams)

    all_mean_and_std = helpers.load_json('../data/all_mean_and_std_{}.json'.format(year))
    stacked_mean_and_std = helpers.load_json('../data/stacked_mean_and_std_{}.json'.format(year))

    within_source_tomorrow_std_shares = standardization_helper_within_source(aggregate_share_within_source_tomorrow,
                                                                             all_mean_and_std, stacked_mean_and_std)
    within_source_in_cluster_std_shares = standardization_helper_within_source(aggregate_share_within_source_in_cluster,
                                                                               all_mean_and_std, stacked_mean_and_std)

    print('calculate bias for within source tomorrow cluster specific std')
    bias_within_source = calculate_bias_group_by_source(within_source_tomorrow_std_shares['specific'], top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_tomorrow_{}_{}_std=specific.csv'.format(year,
                                                                                                                 month))

    print('calculate bias for within source tomorrow cluster overall std')
    bias_within_source = calculate_bias_group_by_source(within_source_tomorrow_std_shares['overall'], top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_tomorrow_{}_{}_std=overall.csv'.format(year,
                                                                                                                month))

    print('calculate bias for within source tomorrow cluster stacked std')
    bias_within_source = calculate_bias_group_by_source(within_source_tomorrow_std_shares['stacked'], top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_tomorrow_{}_{}_std=stacked.csv'.format(year,
                                                                                                                month))

    print('calculate bias for within source in cluster specific std')
    bias_within_source = calculate_bias_group_by_source(within_source_in_cluster_std_shares['specific'], top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_in_cluster_{}_{}_std=stacked.csv'.format(year,
                                                                                                                  month))

    print('calculate bias for within source in cluster overall std')
    bias_within_source = calculate_bias_group_by_source(within_source_in_cluster_std_shares['overall'], top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_in_cluster_{}_{}_std=stacked.csv'.format(year,
                                                                                                                  month))

    print('calculate bias for within source in cluster stacked std')
    bias_within_source = calculate_bias_group_by_source(within_source_in_cluster_std_shares['stacked'], top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_in_cluster_{}_{}_std=stacked.csv'.format(year,
                                                                                                                  month))


def get_aggregate_share_for_year(db_path, dct, tfidf_model, top_bigrams, year, group_by, threshold):
    top_bigrams_share_by_month_in_cluster = []
    total_bigrams_by_month_in_cluster = []
    top_bigrams_share_by_month_not_in_cluster = []
    total_bigrams_by_month_not_in_cluster = []
    top_bigrams_share_by_month_in_cluster_tomorrow = []
    total_bigrams_by_month_in_cluster_tomorrow = []
    top_bigrams_share_by_month_all_articles = []
    total_bigrams_by_month_all_articles = []

    aggregates = {'in_cluster': [], 'not_in_cluster': [], 'in_cluster_tomorrow': [], 'all_articles': []}

    # first get top bigrams shares for all months of the year and also the total count of bigrams for every source
    for month in range(1, 12 + 1):
        bigrams_in_cluster, bigrams_not_in_cluster, bigrams_in_cluster_tomorrow, bigrams_all_articles = \
            get_bigrams_by_clusters(db_path, dct, tfidf_model, year, month, group_by, threshold=threshold)

        # convert bigrams list to bigrams shares
        total_bigrams_all_articles = bigrams.convert_bigrams_to_shares(bigrams_all_articles)
        total_bigrams_in_cluster = bigrams.convert_bigrams_to_shares(bigrams_in_cluster)
        total_bigrams_not_in_cluster = bigrams.convert_bigrams_to_shares(bigrams_not_in_cluster)
        total_bigrams_in_cluster_tomorrow = bigrams.convert_bigrams_to_shares(bigrams_in_cluster_tomorrow)

        # get the shares of top bigrams
        print('get shares of top bigrams in all articles')
        top_bigrams_freq_all_articles = get_shares_of_top1000_bigrams(top_bigrams, bigrams_all_articles)
        print('get shares of top bigrams for in cluster')
        top_bigrams_freq_in_cluster = get_shares_of_top1000_bigrams(top_bigrams, bigrams_in_cluster)
        print('get shares of top bigrams for not in cluster')
        top_bigrams_freq_not_in_cluster = get_shares_of_top1000_bigrams(top_bigrams, bigrams_not_in_cluster)
        print('get shares of top bigrams for in cluster tomorrow')
        top_bigrams_freq_in_cluster_tomorrow = get_shares_of_top1000_bigrams(top_bigrams,
                                                                             bigrams_in_cluster_tomorrow)

        top_bigrams_share_by_month_in_cluster.append(top_bigrams_freq_in_cluster)
        total_bigrams_by_month_in_cluster.append(total_bigrams_in_cluster)

        top_bigrams_share_by_month_not_in_cluster.append(top_bigrams_freq_not_in_cluster)
        total_bigrams_by_month_not_in_cluster.append(total_bigrams_not_in_cluster)

        top_bigrams_share_by_month_in_cluster_tomorrow.append(top_bigrams_freq_in_cluster_tomorrow)
        total_bigrams_by_month_in_cluster_tomorrow.append(total_bigrams_in_cluster_tomorrow)

        top_bigrams_share_by_month_all_articles.append(top_bigrams_freq_all_articles)
        total_bigrams_by_month_all_articles.append(total_bigrams_all_articles)

    # get aggregate bigram count across month, i.e count of bigrams in an year grouped by month
    aggregate_source_count_in_cluster = aggregate_bigrams_month_count(total_bigrams_by_month_in_cluster)
    aggregate_source_count_not_in_cluster = aggregate_bigrams_month_count(total_bigrams_by_month_not_in_cluster)
    aggregate_source_count_in_cluster_tomorrow = aggregate_bigrams_month_count(
        total_bigrams_by_month_in_cluster_tomorrow)
    aggregate_source_count_all_articles = aggregate_bigrams_month_count(total_bigrams_by_month_all_articles)

    # get share of top bigrams for a year by aggregating the share for each month
    print('aggregating bigram share for in cluster')
    aggregate_share_in_cluster = aggregate_bigrams_month_share(top_bigrams_share_by_month_in_cluster,
                                                               total_bigrams_by_month_in_cluster,
                                                               aggregate_source_count_in_cluster,
                                                               top_bigrams)

    print('aggregating bigram share for not in cluster')
    aggregate_share_not_in_cluster = aggregate_bigrams_month_share(top_bigrams_share_by_month_not_in_cluster,
                                                                   total_bigrams_by_month_not_in_cluster,
                                                                   aggregate_source_count_not_in_cluster,
                                                                   top_bigrams)

    print('aggregating bigram share for in cluster tomorrow')
    aggregate_share_in_cluster_tomorrow = aggregate_bigrams_month_share(top_bigrams_share_by_month_in_cluster_tomorrow,
                                                                        total_bigrams_by_month_in_cluster_tomorrow,
                                                                        aggregate_source_count_in_cluster_tomorrow,
                                                                        top_bigrams)

    print('aggregating bigram share for all articles')
    aggregate_share_all_articles = aggregate_bigrams_month_share(top_bigrams_share_by_month_all_articles,
                                                                 total_bigrams_by_month_all_articles,
                                                                 aggregate_source_count_all_articles,
                                                                 top_bigrams)

    aggregates['in_cluster'] = aggregate_share_in_cluster
    aggregates['not_in_cluster'] = aggregate_share_not_in_cluster
    aggregates['in_cluster_tomorrow'] = aggregate_share_in_cluster_tomorrow
    aggregates['all_articles'] = aggregate_share_all_articles
    aggregates['source_count_in_cluster'] = aggregate_source_count_in_cluster
    aggregates['source_count_not_in_cluster'] = aggregate_source_count_not_in_cluster
    aggregates['source_count_in_tomorrow_cluster'] = aggregate_source_count_in_cluster_tomorrow
    aggregates['source_count_all_articles'] = aggregate_source_count_all_articles

    return aggregates


def shares_aggregated_across_year(db_path, dct, tfidf_model, top1000_bigram, group_by, threshold=0.3):
    top_bigrams = top1000_bigram['bigram'].tolist()

    aggregate_2015 = get_aggregate_share_for_year(db_path, dct, tfidf_model, top_bigrams, 2015, group_by, threshold)
    aggregate_2016 = get_aggregate_share_for_year(db_path, dct, tfidf_model, top_bigrams, 2016, group_by, threshold)
    aggregate_2017 = get_aggregate_share_for_year(db_path, dct, tfidf_model, top_bigrams, 2017, group_by, threshold)

    aggregates = [aggregate_2015, aggregate_2016, aggregate_2017]

    top_bigrams_share_in_cluster = [aggregate['in_cluster'] for aggregate in aggregates]
    total_bigrams_count_in_cluster = [aggregate['source_count_in_cluster'] for aggregate in aggregates]
    aggregate_source_count_in_cluster = aggregate_bigrams_month_count(total_bigrams_count_in_cluster)

    aggregate_share_in_cluster = aggregate_bigrams_month_share(top_bigrams_share_in_cluster,
                                                               total_bigrams_count_in_cluster,
                                                               aggregate_source_count_in_cluster,
                                                               top_bigrams)

    top_bigrams_share_not_in_cluster = [aggregate['not_in_cluster'] for aggregate in aggregates]
    total_bigrams_count_not_in_cluster = [aggregate['source_count_not_in_cluster'] for aggregate in aggregates]
    aggregate_source_count_not_in_cluster = aggregate_bigrams_month_count(total_bigrams_count_in_cluster)

    aggregate_share_not_in_cluster = aggregate_bigrams_month_share(top_bigrams_share_not_in_cluster,
                                                                   total_bigrams_count_not_in_cluster,
                                                                   aggregate_source_count_not_in_cluster,
                                                                   top_bigrams)

    top_bigrams_share_all_articles = [aggregate['all_articles'] for aggregate in aggregates]
    total_bigrams_count_all_articles = [aggregate['source_count_all_articles'] for aggregate in aggregates]
    aggregate_source_count_all_articles = aggregate_bigrams_month_count(total_bigrams_count_in_cluster)

    aggregate_share_all_articles = aggregate_bigrams_month_share(top_bigrams_share_all_articles,
                                                                 total_bigrams_count_all_articles,
                                                                 aggregate_source_count_all_articles,
                                                                 top_bigrams)

    mean_and_std = bigrams.get_mean_and_deviation(aggregate_share_all_articles)
    helpers.save_json(mean_and_std, '../data/all_mean_and_std_15_17.json')

    mean_and_std = bigrams.get_stacked_mean_and_deviation(aggregate_share_in_cluster, aggregate_share_not_in_cluster)
    helpers.save_json(mean_and_std, '../data/stacked_mean_and_std_15_17.json')


def bias_averaged_over_year(db_path, dct, tfidf_model, top1000_bigram, year, group_by, threshold=0.3):
    """
    Parameters
    ----------
    @db_path: (string) path to articles database
    @dct: (gensim dictionary object)
    @tfidf_model: (gensim tfidf object)
    @top1000_bigrams: (pandas DataFrame)top 1000 bigrams from MP speeches with alpha and beta bias coefficient
    @year: (int)
    @std_type: type of standardization to apply to top bigrams share
    @threshold: (float)

    Returns
    -------
    None
    """

    assert (1 > threshold > 0)
    top_bigrams = top1000_bigram['bigram'].tolist()

    aggregates = get_aggregate_share_for_year(db_path, dct, tfidf_model, top_bigrams, year, group_by, threshold)
    aggregate_share_all_articles = aggregates['all_articles']
    aggregate_share_in_cluster = aggregates['in_cluster']
    aggregate_share_not_in_cluster = aggregates['not_in_cluster']
    aggregate_share_in_cluster_tomorrow = aggregates['in_cluster_tomorrow']

    aggregate_share_all_articles.to_csv(path_or_buf='../results/bigrams_share_all_articles_{}.csv'.format(year))
    aggregate_share_in_cluster.to_csv(path_or_buf='../results/bigrams_share_in_cluster_{}.csv'.format(year))
    aggregate_share_not_in_cluster.to_csv(path_or_buf='../results/bigrams_share_not_in_cluster_{}.csv'.format(year))
    aggregate_share_in_cluster_tomorrow.to_csv(
        path_or_buf='../results/bigrams_share_in_cluster_tomorrow_{}.csv'.format(year))

    all_mean_and_std = bigrams.get_mean_and_deviation(aggregate_share_all_articles)
    mean_and_std = bigrams.get_stacked_mean_and_deviation(aggregate_share_in_cluster, aggregate_share_not_in_cluster)
    helpers.save_json(mean_and_std, '../data/all_mean_and_std_{}.json'.format(year))
    helpers.save_json(mean_and_std, '../data/stacked_mean_and_std_{}.json'.format(year))

    shares = [aggregate_share_all_articles, aggregate_share_in_cluster, aggregate_share_not_in_cluster,
              aggregate_share_in_cluster_tomorrow]
    standardized_shares = standardization_helper(shares, all_mean_and_std, mean_and_std)
    flatten_std_shares = (standardized_shares['specific'] + standardized_shares['overall'] +
                          standardized_shares['stacked'])

    assert (len(flatten_std_shares) == 12)

    print('calculating bias of news source by cluster groups')
    bias_result = parmap.map(calculate_bias, flatten_std_shares, top1000_bigram, pm_pbar=True)

    assert (len(bias_result) == 12)
    bias_specific_std = bias_result[0:4]
    bias_overall_std = bias_result[4:8]
    bias_stacked_std = bias_result[8:12]

    columns = ['source', 'bias_for_above_median_tomorrow', 'bias_for_below_median_tomorrow',
               'bias_for_above_median_in_cluster', 'bias_for_below_in_cluster']
    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *bias_specific_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_{}_std=specific.csv'.format(year))

    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *bias_overall_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_{}_std=overall.csv'.format(year))

    combined_bias_df = _combine_bias_result_for_all_cluster(columns, *bias_stacked_std)
    combined_bias_df.to_csv(path_or_buf='../results/bias_{}_std=stacked.csv'.format(year))


def main():
    args = parser.parse_args()

    # get dictionary and gensim models
    dct = corpora.Dictionary.load(args.dictionary)
    tfidf_model = models.TfidfModel.load(args.tfidf_model)

    # other config arguments for bias calculation
    year = args.year
    month = args.month
    threshold = args.threshold
    db_path = args.db_path
    top1000_bigrams_path = args.parliament_bigrams
    top_1000_bigrams = pd.read_csv(top1000_bigrams_path)
    group_by = args.group_by
    bias_type = args.bias_type

    if bias_type == 'standardization_parameters':
        shares_aggregated_across_year(db_path, dct, tfidf_model, top_1000_bigrams, group_by, threshold)
    elif bias_type == 'median_groups':
        if month is None:
            bias_averaged_over_year_for_median_clusters(db_path, dct, tfidf_model, top_1000_bigrams, year, group_by,
                                                        threshold)
        else:
            bias_averaged_over_month_for_median_clusters(db_path, dct, tfidf_model, top_1000_bigrams, year, month,
                                                         group_by, threshold)
    elif bias_type == 'within_source':
        if month is None:
            bias_averaged_over_year_for_within_source_clusters(db_path, dct, tfidf_model, top_1000_bigrams, year,
                                                               group_by, threshold)
        else:
            bias_averaged_over_month_for_within_source_clusters(db_path, dct, tfidf_model, top_1000_bigrams, year,
                                                                month, group_by, threshold)
    else:
        if month is None:
            bias_averaged_over_year(db_path, dct, tfidf_model, top_1000_bigrams, year, group_by, threshold=threshold)
        else:
            bias_averaged_over_month(db_path, dct, tfidf_model, top_1000_bigrams, year, month, group_by, threshold)


if __name__ == '__main__':
    main()
