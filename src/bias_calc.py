import argparse
import pandas as pd
from gensim import corpora, models
import cluster_analysis
import parmap
from tqdm import tqdm
import multiprocessing as mp
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
                    choices=['general', 'within_source', 'median_groups'],
                    default='general',
                    help='')

parser.add_argument('-std', '--standardize_type',
                    type=str,
                    choices=['stacked', 'all_articles', 'cluster_specific'],
                    default='cluster_specific',
                    help='user specific standardization type for bigrams shares in clusters')

parser.add_argument('-mt', '--median_type',
                    type=str,
                    choices=['overall', 'source'],
                    default='overall',
                    help='specify whether median should be broken down to source level or overall median should be'
                         'used when calculating median cluster\'s bias')


def get_bigrams_for_median_clusters(db_path, dct, tfidf_model, year, month, group_by, median_type, threshold=0.3):
    """
    A wrapper function to return the bigrams present in the news articles for the given year and month for different
    median clusters.
    Parameters
    ----------
    median_type
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

    above_median, below_median, above_median_in_cluster, below_median_in_cluster = \
        cluster_analysis.get_cluster_of_articles_group_by_median(db_path, dct, tfidf_model, year, month, median_type,
                                                                 threshold)

    print('calculating bigrams_below median')
    bigrams_below_median = bigrams.get_bigrams_in_articles(below_median, group_by)
    print('calculating bigrams above median')
    bigrams_above_median = bigrams.get_bigrams_in_articles(above_median, group_by)
    print('calculating bigrams_below median')
    bigrams_below_median_in_cluster = bigrams.get_bigrams_in_articles(below_median_in_cluster, group_by)
    print('calculating bigrams above median')
    bigrams_above_median_in_cluster = bigrams.get_bigrams_in_articles(above_median_in_cluster, group_by)

    return (bigrams_above_median, bigrams_below_median, bigrams_above_median_in_cluster,
            bigrams_below_median_in_cluster)


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
    bigrams_withon_source_in_cluster:
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

    conn = db.ArticlesDb(db_path)
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
                                                 std_type, median_type, threshold=0.3):
    """

    Parameters
    ----------
    median_type
    db_path
    dct
    tfidf_model
    top1000_bigram
    year
    month
    group_by
    std_type
    threshold

    Returns
    -------

    """
    (bigrams_above_median_tomorrow, bigrams_below_median_tomorrow, bigrams_above_median_in_cluster,
     bigrams_below_median_in_cluster) = get_bigrams_for_median_clusters(db_path, dct, tfidf_model, year, month,
                                                                        group_by, median_type, threshold=threshold)

    top_bigrams = top1000_bigram['bigram'].tolist()

    print('converting bigrams list to fractional count')
    bigrams.convert_bigrams_to_shares(bigrams_above_median_tomorrow)
    bigrams.convert_bigrams_to_shares(bigrams_below_median_tomorrow)
    bigrams.convert_bigrams_to_shares(bigrams_above_median_in_cluster)
    bigrams.convert_bigrams_to_shares(bigrams_below_median_in_cluster)

    print('get top bigrams share for above median tomorrow articles')
    top_bigrams_freq_above_median = get_shares_of_top1000_bigrams(top_bigrams, bigrams_above_median_tomorrow)
    print('get top bigrams share for below median tomorrow articles')
    top_bigrams_freq_below_median = get_shares_of_top1000_bigrams(top_bigrams, bigrams_below_median_tomorrow)
    print('get top bigrams share for above median in cluster articles')
    top_bigrams_freq_above_median2 = get_shares_of_top1000_bigrams(top_bigrams, bigrams_above_median_in_cluster)
    print('get top bigrams share for below median in cluster articles')
    top_bigrams_freq_below_median2 = get_shares_of_top1000_bigrams(top_bigrams, bigrams_below_median_in_cluster)

    del bigrams_above_median_tomorrow, bigrams_below_median_tomorrow, bigrams_above_median_in_cluster, \
        bigrams_below_median_in_cluster

    if std_type == 'cluster_specific':
        print('standardizing bigram count for above median tomorrow articles')
        top_bigrams_freq_above_median = bigrams.standardize_bigrams_count(top_bigrams_freq_above_median)
        print('standardizing bigram count for below median tomorrow articles')
        top_bigrams_freq_below_median = bigrams.standardize_bigrams_count(top_bigrams_freq_below_median)
        print('standardizing bigram count for above median in cluster articles')
        top_bigrams_freq_above_median2 = bigrams.standardize_bigrams_count(top_bigrams_freq_above_median2)
        print('standardizing bigram count for below median in cluster articles')
        top_bigrams_freq_below_median2 = bigrams.standardize_bigrams_count(top_bigrams_freq_below_median2)
    else:
        if std_type == 'all_articles':
            mean_and_std = helpers.load_json('../data/all_mean_and_std_{}_{}.json'.format(year, month))
        elif std_type == 'stacked':
            mean_and_std = helpers.load_json('../data/stacked_mean_and_std_{}_{}.json'.format(year, month))

        print('standardizing bigram count for above median tomorrow articles')
        top_bigrams_freq_above_median = bigrams.standardize_with_mean_and_std(top_bigrams_freq_above_median,
                                                                              mean_and_std)
        print('standardizing bigram count for above median tomorrow articles')
        top_bigrams_freq_below_median = bigrams.standardize_with_mean_and_std(top_bigrams_freq_below_median,
                                                                              mean_and_std)
        print('standardizing bigram count for above median tomorrow articles')
        top_bigrams_freq_above_median2 = bigrams.standardize_with_mean_and_std(top_bigrams_freq_above_median2,
                                                                               mean_and_std)
        print('standardizing bigram count for above median tomorrow articles')
        top_bigrams_freq_below_median2 = bigrams.standardize_with_mean_and_std(top_bigrams_freq_below_median2,
                                                                               mean_and_std)

    print('calculating bias of news source by cluster groups')
    bias_above_median, bias_below_median, bias_above_median2, bias_below_median2 = parmap.map(
        calculate_bias,
        [
            top_bigrams_freq_above_median,
            top_bigrams_freq_below_median,
            top_bigrams_freq_above_median2,
            top_bigrams_freq_below_median2],
        top1000_bigram,
        pm_pbar=True)

    columns = ['source', 'bias_for_above_median_tomorrow', 'bias_for_below_median_tomorrow',
               'bias_for_above_median_in_cluster', 'bias_for_below_in_cluster']
    combined_bias_df = _combine_bias_result_for_all_cluster(columns, bias_above_median, bias_below_median,
                                                            bias_above_median2, bias_below_median2)
    combined_bias_df.to_csv(path_or_buf='../results/bias_{}_median_{}_{}.csv'.format(median_type, year, month))


def bias_averaged_over_month_for_within_source_clusters(db_path, dct, tfidf_model, top1000_bigram, year, month,
                                                        group_by, std_type, threshold=0.3):
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

    if std_type == 'cluster_specific':
        bigrams.standardize_bigrams_count_group_by_source(top_bigrams_freq_within_source_in_cluster)
        bigrams.standardize_with_mean_and_std_group_by_source(top_bigrams_freq_within_source_tomorrow)
    else:
        if std_type == 'all_articles':
            mean_and_std = helpers.load_json('../data/all_mean_and_std_{}_{}.json'.format(year, month))
        elif std_type == 'stacked':
            mean_and_std = helpers.load_json('../data/stacked_mean_and_std_{}_{}.json'.format(year, month))

        print('standardizing bigram count for within source')
        bigrams.standardize_with_mean_and_std_group_by_source(top_bigrams_freq_within_source_tomorrow, mean_and_std)
        print('standardizing bigram count for within source in cluster')
        bigrams.standardize_with_mean_and_std_group_by_source(top_bigrams_freq_within_source_in_cluster,
                                                              mean_and_std)

    print('calculate bias for within source tomorrow cluster')
    bias_within_source = calculate_bias_group_by_source(top_bigrams_freq_within_source_tomorrow, top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_tomorrow_{}_{}.csv'.format(year, month))

    print('calculate bias for within source in cluster')
    bias_within_source = calculate_bias_group_by_source(top_bigrams_freq_within_source_in_cluster, top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_in_cluster_{}_{}.csv'.format(year, month))


def bias_averaged_over_month(db_path, dct, tfidf_model, top1000_bigram, year, month, group_by, std_type, threshold=0.3):
    """
    Parameters
    ----------
    @db_path: (string) path to articles database
    @dct: (gensim dictionary object)
    @tfidf_model: (gensim tfidf object)
    @top1000_bigram: top 1000 bigrams in MP speeches with alpha and beta bias coefficient
    @month: (int)
    @year: (int)
    @bias_type:
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

    if std_type == 'cluster_specific':
        print('standardizing bigram count for all articles')
        top_bigrams_freq_all_articles = bigrams.standardize_with_mean_and_std(top_bigrams_freq_all_articles)
        print('standardizing bigram count in cluster')
        top_bigrams_freq_in_cluster = bigrams.standardize_with_mean_and_std(top_bigrams_freq_in_cluster)
        print('standardizing bigram count not_in cluster')
        top_bigrams_freq_not_in_cluster = bigrams.standardize_with_mean_and_std(top_bigrams_freq_not_in_cluster)
        print('standardizing bigram count for in cluster tomorrow')
        top_bigrams_freq_in_cluster_tomorrow = bigrams.standardize_with_mean_and_std(
            top_bigrams_freq_in_cluster_tomorrow)
    else:
        if std_type == 'all_articles':
            mean_and_std = bigrams.get_mean_and_deviation(top_bigrams_freq_all_articles)
            helpers.save_json(mean_and_std, '../data/all_mean_and_std_{}_{}.json'.format(year, month))
        else:
            mean_and_std = bigrams.get_stacked_mean_and_deviation(top_bigrams_freq_in_cluster,
                                                                  top_bigrams_freq_not_in_cluster)
            helpers.save_json(mean_and_std, '../data/stacked_mean_and_std_{}_{}.json'.format(year, month))

        print('standardizing bigram count for all articles')
        top_bigrams_freq_all_articles = bigrams.standardize_with_mean_and_std(top_bigrams_freq_all_articles,
                                                                              mean_and_std)
        print('standardizing bigram count in cluster')
        top_bigrams_freq_in_cluster = bigrams.standardize_with_mean_and_std(top_bigrams_freq_in_cluster,
                                                                            mean_and_std)
        print('standardizing bigram count not_in cluster')
        top_bigrams_freq_not_in_cluster = bigrams.standardize_with_mean_and_std(top_bigrams_freq_not_in_cluster,
                                                                                mean_and_std)
        print('standardizing bigram count for in cluster tomorrow')
        top_bigrams_freq_in_cluster_tomorrow = bigrams.standardize_with_mean_and_std(
            top_bigrams_freq_in_cluster_tomorrow, mean_and_std)

    print('calculating bias of news source by cluster groups')
    bias_all_articles, bias_in_cluster, bias_not_in_cluster, bias_in_cluster_tomorrow = parmap.map(
        calculate_bias,
        [top_bigrams_freq_all_articles, top_bigrams_freq_in_cluster, top_bigrams_freq_not_in_cluster,
         top_bigrams_freq_in_cluster_tomorrow], top1000_bigram, pm_pbar=True)

    columns = ['source', 'bias_all_articles', 'bias_in_cluster', 'bias_not_in_cluster', 'bias_in_cluster_tomorrow']
    combined_bias_df = _combine_bias_result_for_all_cluster(columns, bias_all_articles, bias_in_cluster,
                                                            bias_not_in_cluster, bias_in_cluster_tomorrow)
    combined_bias_df.to_csv(path_or_buf='../results/bias_{}_{}_{}.csv'.format(year, month, group_by))


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
    assert (len(total_bigrams_for_month) == 12)
    source_across_month = set()

    for i in range(12):
        source_across_month.update(list(total_bigrams_for_month[i].keys()))

    source_across_month = list(source_across_month)

    for source in source_across_month:
        total_bigrams_count[source] = 0

        for i in range(12):
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
    weighted_shares = np.array([0] * 1000, dtype=float)

    if aggregate_source_count[source] == 0:
        row = [source] + list(weighted_shares)
        return row

    for i in range(12):
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

    assert (len(top_bigrams_month_share) == 12)
    assert (len(total_bigrams_month) == 12)
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
    for source in tqdm(helpers.source_names):
        bigrams_month_share = [top_bigrams_month_share[i][source] for i in range(12)]
        bigrams_month_count = [total_bigrams_month[i][source] for i in range(12)]
        aggregate_bigrams_share[source] = aggregate_bigrams_month_share(bigrams_month_share, bigrams_month_count,
                                                                        aggregate_source_count[source], top_bigrams,
                                                                        pbar=False)

    return aggregate_bigrams_share


def bias_averaged_over_year_for_median_clusters(db_path, dct, tfidf_model, top1000_bigram, year, group_by, std_type,
                                                median_type, threshold=0.3):
    assert (1 > threshold > 0)
    top_bigrams = top1000_bigram['bigram'].tolist()

    top_bigrams_share_by_month_above_median = []
    total_bigrams_by_month_above_median = []
    top_bigrams_share_by_month_below_median = []
    total_bigrams_by_month_below_median = []
    top_bigrams_share_by_month_above_median_in_cluster = []
    total_bigrams_by_month_above_median_in_cluster = []
    top_bigrams_share_by_month_below_median_in_cluster = []
    total_bigrams_by_month_below_median_in_cluster = []

    # first get top bigrams shares for all months of the year and also the total count of bigrams for every source
    for month in range(1, 12 + 1):
        (bigrams_above_median_tomorrow, bigrams_below_median_tomorrow, bigrams_above_median_in_cluster,
         bigrams_below_median_in_cluster) = get_bigrams_for_median_clusters(db_path, dct, tfidf_model, year, month,
                                                                            group_by, median_type, threshold=threshold)

        # convert bigrams list to bigrams shares
        total_bigrams_above_median_tomorrow = bigrams.convert_bigrams_to_shares(bigrams_above_median_tomorrow)
        total_bigrams_below_median_tomorrow = bigrams.convert_bigrams_to_shares(bigrams_below_median_tomorrow)
        total_bigrams_above_median_in_cluster = bigrams.convert_bigrams_to_shares(bigrams_above_median_in_cluster)
        total_bigrams_below_median_in_cluster = bigrams.convert_bigrams_to_shares(bigrams_below_median_in_cluster)

        # get the shares of top bigrams
        print('get shares of top bigrams in above median tomorrow articles')
        top_bigrams_freq_above_median = get_shares_of_top1000_bigrams(top_bigrams, bigrams_above_median_tomorrow)
        print('get shares of top bigrams for below median tomorrow articles')
        top_bigrams_freq_below_median = get_shares_of_top1000_bigrams(top_bigrams, bigrams_below_median_tomorrow)
        print('get shares of top bigrams in above median in cluster articles')
        top_bigrams_freq_above_median2 = get_shares_of_top1000_bigrams(top_bigrams, bigrams_above_median_in_cluster)
        print('get shares of top bigrams for below median in cluster articles')
        top_bigrams_freq_below_median2 = get_shares_of_top1000_bigrams(top_bigrams, bigrams_below_median_in_cluster)

        top_bigrams_share_by_month_above_median.append(top_bigrams_freq_above_median)
        total_bigrams_by_month_above_median.append(total_bigrams_above_median_tomorrow)

        top_bigrams_share_by_month_below_median.append(top_bigrams_freq_below_median)
        total_bigrams_by_month_below_median.append(total_bigrams_below_median_tomorrow)

        top_bigrams_share_by_month_above_median_in_cluster.append(top_bigrams_freq_above_median2)
        total_bigrams_by_month_above_median_in_cluster.append(total_bigrams_above_median_in_cluster)

        top_bigrams_share_by_month_below_median_in_cluster.append(top_bigrams_freq_below_median2)
        total_bigrams_by_month_below_median_in_cluster.append(total_bigrams_below_median_in_cluster)

    # get aggregate bigram count across month, i.e count of bigrams in an year grouped by month
    aggregate_source_count_above_median = aggregate_bigrams_month_count(total_bigrams_by_month_above_median)
    aggregate_source_count_below_median = aggregate_bigrams_month_count(total_bigrams_by_month_below_median)
    aggregate_source_count_above_median2 = aggregate_bigrams_month_count(total_bigrams_by_month_above_median_in_cluster)
    aggregate_source_count_below_median2 = aggregate_bigrams_month_count(total_bigrams_by_month_below_median_in_cluster)

    # get share of top bigrams for a year by aggregating the share for each month
    print('aggregating bigram share for above median cluster')
    aggregate_share_above_median = aggregate_bigrams_month_share(top_bigrams_share_by_month_above_median,
                                                                 total_bigrams_by_month_above_median,
                                                                 aggregate_source_count_above_median,
                                                                 top_bigrams)

    print('aggregating bigram share for below median cluster')
    aggregate_share_below_median = aggregate_bigrams_month_share(top_bigrams_share_by_month_below_median,
                                                                 total_bigrams_by_month_below_median,
                                                                 aggregate_source_count_below_median,
                                                                 top_bigrams)

    print('aggregating bigram share for above median cluster')
    aggregate_share_above_median2 = aggregate_bigrams_month_share(
        top_bigrams_share_by_month_above_median_in_cluster,
        total_bigrams_by_month_above_median_in_cluster,
        aggregate_source_count_above_median2,
        top_bigrams)

    print('aggregating bigram share for below median cluster')
    aggregate_share_below_median2 = aggregate_bigrams_month_share(
        top_bigrams_share_by_month_below_median_in_cluster,
        total_bigrams_by_month_below_median_in_cluster,
        aggregate_source_count_below_median2,
        top_bigrams)

    if std_type == 'cluster_specific':
        print('standardizing bigram count for above median tomorrow articles')
        aggregate_share_above_median = bigrams.standardize_bigrams_count(aggregate_share_above_median)
        print('standardizing bigram count below median tomorrow articles')
        aggregate_share_below_median = bigrams.standardize_bigrams_count(aggregate_share_below_median)
        print('standardizing bigram count for above median in cluster articles')
        aggregate_share_above_median2 = bigrams.standardize_bigrams_count(aggregate_share_above_median2)
        print('standardizing bigram count below median in cluster articles')
        aggregate_share_below_median2 = bigrams.standardize_bigrams_count(aggregate_share_below_median2)
    else:
        if std_type == 'all_articles':
            mean_and_std = helpers.load_json('../data/all_mean_and_std_{}.json'.format(year))
        elif std_type == 'stacked':
            mean_and_std = helpers.load_json('../data/stacked_mean_and_std_{}.json'.format(year))

        print('standardizing bigram count for above median tomorrow articles')
        aggregate_share_above_median = bigrams.standardize_with_mean_and_std(aggregate_share_above_median,
                                                                             mean_and_std)
        print('standardizing bigram count for above median tomorrow articles')
        aggregate_share_below_median = bigrams.standardize_with_mean_and_std(aggregate_share_below_median,
                                                                             mean_and_std)
        print('standardizing bigram count for above median tomorrow articles')
        aggregate_share_above_median2 = bigrams.standardize_with_mean_and_std(aggregate_share_above_median2,
                                                                              mean_and_std)
        print('standardizing bigram count for above median tomorrow articles')
        aggregate_share_below_median2 = bigrams.standardize_with_mean_and_std(aggregate_share_below_median2,
                                                                              mean_and_std)

    print('calculating bias of news source by cluster groups')
    bias_above_median, bias_below_median, bias_above_median2, bias_below_median2 = parmap.map(
        calculate_bias,
        [
            aggregate_share_above_median,
            aggregate_share_below_median,
            aggregate_share_above_median2,
            aggregate_share_below_median2],
        top1000_bigram)

    columns = ['source', 'bias_above_median_tomorrow', 'bias_below_median_tomorrow', 'bias_above_median_in_cluster',
               'bias_below_median_in_cluster']
    combined_bias_df = _combine_bias_result_for_all_cluster(columns, bias_above_median, bias_below_median,
                                                            bias_above_median2, bias_below_median2)
    combined_bias_df.to_csv(path_or_buf='../results/bias_median_{}.csv'.format(year))


def bias_averaged_over_year_for_within_source_clusters(db_path, dct, tfidf_model, top1000_bigram, year, group_by,
                                                       std_type, threshold=0.3):
    assert (1 > threshold > 0)
    top_bigrams = top1000_bigram['bigram'].tolist()

    top_bigrams_share_by_month_within_source_tomorrow = []
    total_bigrams_by_month_within_source_tomorrow = []
    top_bigrams_share_by_month_within_source_in_cluster = []
    total_bigrams_by_month_within_source_in_cluster = []
    for month in range(1, 12 + 1):
        bigrams_within_source_tomorrow, bigrams_within_source_in_cluster = \
            get_bigrams_for_within_source_clusters(db_path, dct, tfidf_model, year, month, group_by, bias_type,
                                                   threshold)

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

    if std_type == 'cluster_specific':
        print('standardizing bigrams count for within source tomorrow')
        bigrams.standardize_bigrams_count_group_by_source(aggregate_share_within_source_tomorrow)
        print('standardizing bigrams count for within source in cluster')
        bigrams.standardize_bigrams_count_group_by_source(aggregate_share_within_source_in_cluster)
    else:
        if std_type == 'all_articles':
            mean_and_std = helpers.load_json('../data/all_mean_and_std_{}.json'.format(year))
        elif std_type == 'stacked':
            mean_and_std = helpers.load_json('../data/stacked_mean_and_std_{}.json'.format(year))

        print('standardizing bigram count for within source')
        bigrams.standardize_with_mean_and_std_group_by_source(aggregate_share_within_source_tomorrow, mean_and_std)
        print('standardizing bigram count for within source in cluster')
        bigrams.standardize_with_mean_and_std_group_by_source(aggregate_share_within_source_in_cluster,
                                                              mean_and_std)

    print('calculating bias within source tomorrow')
    bias_within_source = calculate_bias_group_by_source(aggregate_share_within_source_tomorrow, top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_tomorrow_{}.csv'.format(year))

    print('calculating bias within source in cluster')
    bias_within_source = calculate_bias_group_by_source(aggregate_share_within_source_in_cluster, top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_in_cluster_{}.csv'.format(year))


def bias_averaged_over_year(db_path, dct, tfidf_model, top1000_bigram, year, group_by, bias_type, std_type,
                            threshold=0.3):
    """
    Parameters
    ----------
    @db_path: (string) path to articles database
    @dct: (gensim dictionary object)
    @tfidf_model: (gensim tfidf object)
    @top1000_bigrams: (pandas DataFrame)top 1000 bigrams from MP speeches with alpha and beta bias coefficient
    @year: (int)
    @bias_type: type of bias, whether within source bias, cluster bias or median bias
    @std_type: type of standardization to apply to top bigrams share
    @threshold: (float)

    Returns
    -------
    None
    """

    assert (1 > threshold > 0)
    top_bigrams = top1000_bigram['bigram'].tolist()

    top_bigrams_share_by_month_in_cluster = []
    total_bigrams_by_month_in_cluster = []
    top_bigrams_share_by_month_not_in_cluster = []
    total_bigrams_by_month_not_in_cluster = []
    top_bigrams_share_by_month_in_cluster_tomorrow = []
    total_bigrams_by_month_in_cluster_tomorrow = []
    top_bigrams_share_by_month_all_articles = []
    total_bigrams_by_month_all_articles = []

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
    aggregate_share_in_cluster_tomorrow = aggregate_bigrams_month_share(
        top_bigrams_share_by_month_in_cluster_tomorrow,
        total_bigrams_by_month_in_cluster_tomorrow,
        aggregate_source_count_in_cluster_tomorrow,
        top_bigrams)

    print('aggregating bigram share for all articles')
    aggregate_share_all_articles = aggregate_bigrams_month_share(top_bigrams_share_by_month_all_articles,
                                                                 total_bigrams_by_month_all_articles,
                                                                 aggregate_source_count_all_articles,
                                                                 top_bigrams)

    aggregate_share_all_articles.to_csv(path_or_buf='../results/bigrams_share_all_articles_{}.csv'.format(year))
    aggregate_share_in_cluster.to_csv(path_or_buf='../results/bigrams_share_in_cluster_{}.csv'.format(year))
    aggregate_share_not_in_cluster.to_csv(path_or_buf='../results/bigrams_share_not_in_cluster_{}.csv'.format(year))
    aggregate_share_in_cluster_tomorrow.to_csv(
        path_or_buf='../results/bigrams_share_in_cluster_tomorrow_{}.csv'.format(year))

    helpers.save_json(aggregate_source_count_all_articles, '../results/total_bigrams_all_articles_{}.json'.format(
        year))
    helpers.save_json(aggregate_source_count_in_cluster, '../results/total_bigrams_in_cluster_{}.json'.format(year))
    helpers.save_json(aggregate_source_count_not_in_cluster, '../results/total_bigrams_not_in_cluster_{}.json'.
                      format(year))
    helpers.save_json(aggregate_source_count_in_cluster_tomorrow,
                      '../results/total_bigrams_in_cluster_tomorrow_{}.json'.format(year))

    if std_type == 'cluster_specific':
        print('standardizing bigram count for all articles')
        aggregate_share_in_cluster = bigrams.standardize_bigrams_count(aggregate_share_in_cluster)
        print('standardizing bigram count in cluster')
        aggregate_share_not_in_cluster = bigrams.standardize_bigrams_count(aggregate_share_not_in_cluster)
        print('standardizing bigram count not_in cluster')
        aggregate_share_in_cluster_tomorrow = bigrams.standardize_bigrams_count(aggregate_share_in_cluster_tomorrow)
        print('standardizing bigram count for in cluster tomorrow')
        aggregate_share_all_articles = bigrams.standardize_bigrams_count(aggregate_share_all_articles)
    else:
        if std_type == 'all_articles':
            mean_and_std = bigrams.get_mean_and_deviation(aggregate_share_all_articles)
            helpers.save_json(mean_and_std, '../data/all_mean_and_std_{}.json'.format(year))
        else:
            mean_and_std = bigrams.get_stacked_mean_and_deviation(aggregate_share_in_cluster,
                                                                  aggregate_share_not_in_cluster)
            helpers.save_json(mean_and_std, '../data/stacked_mean_and_std_{}.json'.format(year))

        print('standardizing bigram count for all articles')
        aggregate_share_all_articles = bigrams.standardize_with_mean_and_std(aggregate_share_all_articles,
                                                                             mean_and_std)
        print('standardizing bigram count in cluster')
        aggregate_share_in_cluster = bigrams.standardize_with_mean_and_std(aggregate_share_in_cluster,
                                                                           mean_and_std)
        print('standardizing bigram count not_in cluster')
        aggregate_share_not_in_cluster = bigrams.standardize_with_mean_and_std(aggregate_share_not_in_cluster,
                                                                               mean_and_std)
        print('standardizing bigram count for in cluster tomorrow')
        aggregate_share_in_cluster_tomorrow = bigrams.standardize_with_mean_and_std(
            aggregate_share_in_cluster_tomorrow, mean_and_std)

    print('calculating bias of news source by cluster groups')
    bias_all_articles, bias_in_cluster, bias_not_in_cluster, bias_in_cluster_tomorrow = parmap.map(
        calculate_bias,
        [
            aggregate_share_all_articles,
            aggregate_share_in_cluster,
            aggregate_share_not_in_cluster,
            aggregate_share_in_cluster_tomorrow],
        top1000_bigram)

    columns = ['source', 'bias_all_articles', 'bias_in_cluster', 'bias_not_in_cluster', 'bias_in_cluster_tomorrow']
    combined_bias_df = _combine_bias_result_for_all_cluster(columns, bias_all_articles, bias_in_cluster,
                                                            bias_not_in_cluster, bias_in_cluster_tomorrow)
    combined_bias_df.to_csv(path_or_buf='../results/bias_{}_{}.csv'.format(year, group_by))


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
    std_type = args.standardize_type

    if bias_type == 'median_groups':
        median_type = args.median_type
        if month is None:
            bias_averaged_over_year_for_median_clusters(db_path, dct, tfidf_model, top_1000_bigrams, year, group_by,
                                                        std_type, median_type, threshold)
        else:
            bias_averaged_over_month_for_median_clusters(db_path, dct, tfidf_model, top_1000_bigrams, year, month,
                                                         group_by, std_type, median_type, threshold)
    elif bias_type == 'within_source':
        if month is None:
            bias_averaged_over_year_for_within_source_clusters(db_path, dct, tfidf_model, top_1000_bigrams, year,
                                                               group_by, std_type, threshold)
        else:
            bias_averaged_over_month_for_within_source_clusters(db_path, dct, tfidf_model, top_1000_bigrams, year,
                                                                month, group_by, std_type, threshold)
    else:
        if month is None:
            bias_averaged_over_year(db_path, dct, tfidf_model, top_1000_bigrams, year, group_by, std_type,
                                    threshold=threshold)
        else:
            bias_averaged_over_month(db_path, dct, tfidf_model, top_1000_bigrams, year, month, group_by, std_type,
                                     threshold=threshold)


if __name__ == '__main__':
    main()
