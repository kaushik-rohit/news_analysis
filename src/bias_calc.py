import argparse
import calendar
from datetime import date, timedelta
import pandas as pd
import itertools
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from gensim import corpora, models
from cluster_analysis import get_articles_not_in_cluster, get_similar_articles
import parmap
from collections import Counter
from tqdm import tqdm
import numpy as np
import db
import string
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

porter = PorterStemmer()
source_names = ['Sun', 'Mirror', 'Belfast Telegraph', 'Record', 'Independent', 'Observer', 'Guardian', 'People',
                'Telegraph', 'Mail', 'Express', 'Post', 'Herald', 'Star', 'Wales', 'Scotland', 'Standard', 'Scotsman']

source_ids = ['400553', '377101', '418973', '244365', '8200', '412338', '138794', '232241', '334988', '331369',
              '138620', '419001', '8010', '142728', '408506', '143296', '363952', '145251', '232240', '145253',
              '389195', '145254', '344305', '8109', '397135', '163795', '412334', '408508', '411938']


def ngrams_wrapper(sent):
    return list(nltk.ngrams(sent, 2))


def stem_wrapper(sent):
    return list(map(porter.stem, sent))


def preprocess(sent):
    """
    remove stop words and punctuations from tokens in sentence
    Parameters
    ----------
    sent: tokenized sentence, list of word tokens

    Returns
    -------
    filtered list of sentence tokens
    """
    stop_words = set(stopwords.words('english'))
    return list(filter(lambda token: token not in string.punctuation and token not in stop_words and (
            len(token) > 1), sent))


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
    conn = db.ArticlesDb(path)

    within_source_cluster = {source: [] for source in source_names}

    articles_day1 = list(conn.select_articles_by_date(curr_date))
    articles_day2 = list(conn.select_articles_by_date(next_date))

    unclustered_articles_indices = get_articles_not_in_cluster(articles_day1, dct, tfidf_model, threshold=threshold)
    unclustered_articles = [articles_day1[i] for i in unclustered_articles_indices]
    clustered_articles = [articles_day1[i] for i in range(len(articles_day1)) if i not in unclustered_articles_indices]
    unclustered_articles_indices_in_day2_cluster = get_similar_articles(unclustered_articles, articles_day2, dct,
                                                                        tfidf_model, threshold=threshold)
    unclustered_articles_in_day2_cluster = [unclustered_articles[i] for i, idx in
                                            unclustered_articles_indices_in_day2_cluster]

    for idx, indices in unclustered_articles_indices_in_day2_cluster:
        source = unclustered_articles[idx].source
        for i in indices:
            within_source_cluster[source].append(articles_day2[i])

    return clustered_articles, unclustered_articles, unclustered_articles_in_day2_cluster, within_source_cluster


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
    within_cluster_source = {source: [] for source in source_names}

    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    print('calculating clusters for {} {}'.format(year, calendar.month_name[month]))
    stats = parmap.map(get_cluster_for_the_day, date_range, path, dct, tfidf_model, threshold, pm_pbar=True)

    for stat in stats:
        in_cluster_articles += stat[0]
        not_in_cluster_articles += stat[1]
        not_in_cluster_but_next_day_cluster += stat[2]
        within_cluster_source = helpers.combine_dct(within_cluster_source, stat[3])

    return in_cluster_articles, not_in_cluster_articles, not_in_cluster_but_next_day_cluster, within_cluster_source


def get_bigrams_for_single_article(article, group_by):
    """
    Calculates bigrams present in a news article
    Parameters
    ----------
    @article: Article Object

    Returns
    -------
    (source, bigram) source name and list of bigrams in article transcript as a tuple
    """

    sentences = nltk.sent_tokenize(article.transcript.lower())
    tokenized = map(nltk.tokenize.word_tokenize, sentences)
    tokenized = map(preprocess, tokenized)
    tokenized = map(stem_wrapper, tokenized)
    bigrams = map(ngrams_wrapper, tokenized)
    bigram = list(itertools.chain.from_iterable(bigrams))

    for i in range(len(bigram)):
        bigram[i] = bigram[i][0] + '_' + bigram[i][1]

    if group_by == 'source_id':
        return article.source_id, bigram

    return article.source, bigram


def get_bigrams_in_articles(articles, group_by):
    """
    Calculates bigrams present in a list of news articles. These bigrams are combined and grouped by news source.
    It calculates the bigram for articles in parallel and later combine these bigrams to form a dictionary.
    The method return a dictionary with keys as news source and values as bigrams.
    Parameters
    ----------
    @articles: list of news articles

    Returns
    -------
    A dictionary with key as source and value as a Counter object containing frequency of bigrams in the articles from
    these news source
    """

    all_bigrams = {source: [] for source in source_names}

    bigrams_by_source = parmap.map(get_bigrams_for_single_article, articles, group_by, pm_pbar=True)

    # for _ in tqdm(pool.imap_unordered(get_bigrams_for_single_article, articles), total=len(articles)):
    #     bigrams_by_source.append(_)

    for source, bigrams in bigrams_by_source:
        all_bigrams[source] += bigrams

    return all_bigrams


def _convert_bigrams_to_shares(all_bigrams):
    """
    Converts list of bigrams to a dictionary with keys as unique bigram and value as the share of bigram over all the
    bigrams present for that source.

    Parameters
    ----------
    all_bigrams: (dictionary) A python dictionary with key as news source and value as list of all bigrams present
    in the news of the news source

    Returns
    -------
    total_count_by_source (dictionary): with count of all the bigrams for every source i.e key is source and value is
    the count of bigrams in that source
    """

    total_count_by_source = {}

    for source, bigrams in all_bigrams.items():
        bigrams_freq = Counter(bigrams)
        total = sum(bigrams_freq.values(), 0.0)
        total_count_by_source[source] = total

        for bigram in bigrams_freq:
            bigrams_freq[bigram] /= total

        all_bigrams[source] = bigrams_freq

    return total_count_by_source


def get_bigrams_for_year_and_month_by_clusters(db_path, dct, tfidf_model, year, month, group_by, threshold=0.3):
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
    """
    conn = db.ArticlesDb(db_path)
    news_source = conn.get_news_source_for_month(year, month)
    all_articles = list(conn.select_articles_by_year_and_month(year, month))
    n_articles = conn.get_count_of_articles_for_year_and_month(year, month)

    assert (n_articles == len(all_articles))

    in_cluster, not_in_cluster, in_cluster_tomorrow, within_source_cluster = get_cluster_of_articles(db_path,
                                                                                                     dct, tfidf_model,
                                                                                                     year, month,
                                                                                                     threshold)

    assert (n_articles == (len(in_cluster) + len(not_in_cluster)))

    bigrams_within_source_cluster = {}

    for key, val in within_source_cluster.items():
        print('calculating bigrams for within cluster source articles {}'.format(key))
        bigrams_within_source_cluster[key] = get_bigrams_in_articles(val, group_by)

    print('calculating bigrams for in cluster articles')
    bigrams_in_cluster = get_bigrams_in_articles(in_cluster, group_by)
    print('calculating bigrams for not in cluster articles')
    bigrams_not_in_cluster = get_bigrams_in_articles(not_in_cluster, group_by)
    print('calculating bigrams for in tomorrow cluster articles')
    bigrams_in_cluster_tomorrow = get_bigrams_in_articles(in_cluster_tomorrow, group_by)
    print('calculating bigrams for all articles')
    bigrams_all_articles = get_bigrams_in_articles(all_articles, group_by)

    return (bigrams_in_cluster, bigrams_not_in_cluster, bigrams_in_cluster_tomorrow, bigrams_all_articles,
            bigrams_within_source_cluster)


def standardize_bigrams_count(top_bigrams_share_by_source):
    """
    Calculate Z-Score for shares of 1000 bigrams from MP Speeches in articles grouped by source.
    Update the bigram share and standardize it.
    Parameters
    ----------
    top_bigrams_share_by_source: (pandas DataFrame) with row as share of top bigrams in each source and columns as
                                the top bigrams
    Returns
    -------
    the pandas DataFrame with the standardized shares
    """

    bigram_set = top_bigrams_share_by_source.columns.tolist()[1:]

    assert (len(bigram_set) == 1000)

    for bigram in tqdm(bigram_set):
        std = top_bigrams_share_by_source[bigram].std()
        mean = top_bigrams_share_by_source[bigram].mean()
        top_bigrams_share_by_source[bigram] = (top_bigrams_share_by_source[bigram] - mean) / std

    return top_bigrams_share_by_source.fillna(0)  # if z score is nan change it 0


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


def get_shares_of_top1000_bigrams(top1000_bigram, bigrams):
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

    rows = parmap.starmap(get_shares_of_top_1000_bigrams_for_source, bigrams.items(), top1000_bigram, pm_pbar=True)

    columns = ['source'] + top1000_bigram

    return pd.DataFrame(rows, columns=columns).sort_values(by=['source']).reset_index(drop=True)


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

    bigrams = top1000_bigrams_freq_by_source.columns.tolist()[1:]

    assert (len(bigrams) == 1000)

    bias_by_source = {}

    for index, row in tqdm(top1000_bigrams_freq_by_source.iterrows()):
        num = 0
        den = 0
        # print('count of source {} is {} '.format(row['source'], bigrams_count_by_source[row['source']]))
        for i in range(1000):
            alpha = top1000bigrams[top1000bigrams['bigram'] == bigrams[i]].iloc[0]['alpha']
            beta = top1000bigrams[top1000bigrams['bigram'] == bigrams[i]].iloc[0]['beta']
            bigram_share = row[bigrams[i]]

            assert (isinstance(alpha, float))
            assert (isinstance(beta, float))
            assert (isinstance(bigram_share, float))

            # print('alpha: {}, beta: {}'.format(alpha, beta))
            num += beta * (bigram_share - alpha)
            den += beta * beta
        # print('numerator: {}, denominator: {}'.format(num, den))
        bias = num / den
        bias_by_source[row['source']] = bias

    return bias_by_source


def _combine_bias_result_for_all_cluster(all_articles, in_cluster, not_in_cluster, in_tomorrow_cluster):
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

    columns = ['source', 'all_articles', 'in_cluster', 'not_in_cluster', 'in_tomorrows_cluster']
    sources = (list(all_articles.keys()) + list(in_cluster.keys()) + list(not_in_cluster.keys()) +
               list(in_tomorrow_cluster.keys()))

    sources = list(set(sources))
    rows = []
    for source in sources:
        bias_all_articles = all_articles[source] if source in all_articles else 0
        bias_in_cluster = in_cluster[source] if source in in_cluster else 0
        bias_not_in_cluster = not_in_cluster[source] if source in not_in_cluster else 0
        bias_in_tomorrow_cluster = in_tomorrow_cluster[source] if source in in_tomorrow_cluster else 0

        rows += [[source, bias_all_articles, bias_in_cluster, bias_not_in_cluster, bias_in_tomorrow_cluster]]

    return pd.DataFrame(rows, columns=columns).sort_values(by=['source']).reset_index(drop=True)


def calculate_bias_for_within_source_cluster(cluster, top1000_bigram):
    rows = []
    columns = ['source'] + source_names
    top_bigrams = top1000_bigram['bigram'].tolist()

    for source, bigrams in cluster.items():
        _convert_bigrams_to_shares(bigrams)
        top_bigrams_freq = get_shares_of_top1000_bigrams(top_bigrams, bigrams)
        top_bigrams_freq = standardize_bigrams_count(top_bigrams_freq)
        bias = calculate_bias(top_bigrams_freq, top1000_bigram)
        row = [source]
        for source_name in source_names:
            row += [bias[source_name]]
        rows += [row]

    return pd.DataFrame(rows, columns=columns).sort_values(by=['source']).reset_index(drop=True)


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
    @agg_later: (boolean) whether the bias are aggregated later (in bias_averaged_over_year)
    @threshold: (float)

    Returns
    -------
    if agg_later=False:
        a pandas dataframe with bias for different clusters grouped by news source
    else:
        dictionary objects for different clusters which has source as key and corresponding month bias.
        This is used when yearly bias for source is to be calculated and hence the results are aggregated at later
        stage for each of the months in a year.
    """

    (bigrams_in_cluster, bigrams_not_in_cluster, bigrams_in_cluster_tomorrow, bigrams_all_articles,
     bigrams_within_source) = get_bigrams_for_year_and_month_by_clusters(db_path, dct,
                                                                         tfidf_model, year,
                                                                         month, group_by, threshold)

    # print(bigrams_within_source)
    print('calculating bias for within source cluster')
    bias_within_source = calculate_bias_for_within_source_cluster(bigrams_within_source, top1000_bigram)
    bias_within_source.to_csv(path_or_buf='../results/bias_within_source_{}_{}_{}.csv'.format(year, month, group_by))

    print('converting bigrams list to fractional count')
    _convert_bigrams_to_shares(bigrams_all_articles)
    _convert_bigrams_to_shares(bigrams_in_cluster)
    _convert_bigrams_to_shares(bigrams_not_in_cluster)
    _convert_bigrams_to_shares(bigrams_in_cluster_tomorrow)

    top_bigrams = top1000_bigram['bigram'].tolist()

    print('get top bigrams share for all articles')
    top_bigrams_freq_all_articles = get_shares_of_top1000_bigrams(top_bigrams, bigrams_all_articles)
    print('get top bigrams share for in cluster')
    top_bigrams_freq_in_cluster = get_shares_of_top1000_bigrams(top_bigrams, bigrams_in_cluster)
    print('get top bigrams for not in cluster')
    top_bigrams_freq_not_in_cluster = get_shares_of_top1000_bigrams(top_bigrams, bigrams_not_in_cluster)
    print('get top bigrams for in cluster tomorrow')
    top_bigrams_freq_in_cluster_tomorrow = get_shares_of_top1000_bigrams(top_bigrams, bigrams_in_cluster_tomorrow)

    # top_bigrams_freq_all_articles.to_csv(path_or_buf='../results/top_bigrams_freq_all_articles_{}_{}.csv'.format(year,
    #                                                                                                            month))
    # top_bigrams_freq_in_cluster.to_csv(path_or_buf='../results/top_bigrams_freq_in_cluster_{}_{}.csv'.format(year,
    #                                                                                                          month))
    # top_bigrams_freq_not_in_cluster.to_csv(path_or_buf='../results/top_bigrams_freq_not_in_cluster_{}_{}.csv'.format(
    #     year, month))
    # top_bigrams_freq_in_cluster_tomorrow.to_csv(path_or_buf='../results/top_bigrams_freq_in_cluster_tomorrow_{}_{}.csv'
    #                                             .format(year, month))

    del bigrams_in_cluster, bigrams_in_cluster_tomorrow, bigrams_not_in_cluster, bigrams_all_articles

    print('standardizing bigram count for all articles')
    top_bigrams_freq_all_articles = standardize_bigrams_count(top_bigrams_freq_all_articles)
    print('standardizing bigram count in cluster')
    top_bigrams_freq_in_cluster = standardize_bigrams_count(top_bigrams_freq_in_cluster)
    print('standardizing bigram count not_in cluster')
    top_bigrams_freq_not_in_cluster = standardize_bigrams_count(top_bigrams_freq_not_in_cluster)
    print('standardizing bigram count for in cluster tomorrow')
    top_bigrams_freq_in_cluster_tomorrow = standardize_bigrams_count(top_bigrams_freq_in_cluster_tomorrow)

    # top_bigrams_freq_all_articles.to_csv(path_or_buf='../results/top_bigrams_freq_all_articles_std_{}_{}.csv'.format(
    #     year, month))
    # top_bigrams_freq_in_cluster.to_csv(path_or_buf='../results/top_bigrams_freq_in_cluster_std_{}_{}.csv'.format(year,
    #                                                                                                              month))
    # top_bigrams_freq_not_in_cluster.to_csv(path_or_buf='../results/top_bigrams_freq_not_in_cluster_std_{}_{}.csv'.
    #                                        format(year, month))
    # top_bigrams_freq_in_cluster_tomorrow.to_csv(
    #     path_or_buf='../results/top_bigrams_freq_in_cluster_tomorrow_std_{}_{}.csv'.format(year, month))

    print('calculating bias of news source by cluster groups')
    bias_all_articles, bias_in_cluster, bias_not_in_cluster, bias_in_cluster_tomorrow = parmap.map(
        calculate_bias, [top_bigrams_freq_all_articles, top_bigrams_freq_in_cluster,
                         top_bigrams_freq_not_in_cluster, top_bigrams_freq_in_cluster_tomorrow], top1000_bigram)

    print(bias_all_articles)

    return _combine_bias_result_for_all_cluster(bias_all_articles, bias_in_cluster, bias_not_in_cluster,
                                                bias_in_cluster_tomorrow)


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


def aggregate_bigrams_month_share_for_source(source, top_bigrams_month_share, total_bigrams_month,
                                             aggregate_source_count):
    weighted_shares = np.array([0] * 1000, dtype=float)
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


def aggregate_bigrams_month_share(top_bigrams_month_share, total_bigrams_month, aggregate_source_count, top_bigrams):
    """

    Parameters
    ----------
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
                      aggregate_source_count, pm_pbar=True)

    return pd.DataFrame(rows, columns=columns).sort_values(by=['source']).reset_index(drop=True)


def bias_averaged_over_year(db_path, dct, tfidf_model, top1000_bigram, year, group_by, threshold=0.3):
    """
    Parameters
    ----------
    @db_path: (string) path to articles database
    @dct: (gensim dictionary object)
    @tfidf_model: (gensim tfidf object)
    @top1000_bigrams: (pandas DataFrame)top 1000 bigrams from MP speeches with alpha and beta bias coefficient
    @year: (int)
    @threshold: (float)

    Returns
    -------
    a pandas dataframe with bias for different clusters grouped by source
    """

    assert (1 > threshold > 0)
    top_bigrams_share_by_month_in_cluster = []
    total_bigrams_by_month_in_cluster = []
    top_bigrams_share_by_month_not_in_cluster = []
    total_bigrams_by_month_not_in_cluster = []
    top_bigrams_share_by_month_in_cluster_tomorrow = []
    total_bigrams_by_month_in_cluster_tomorrow = []
    top_bigrams_share_by_month_all_articles = []
    total_bigrams_by_month_all_articles = []

    top_bigrams = top1000_bigram['bigram'].tolist()

    # first get top bigrams shares for all months of the year and also the total count of bigrams for every source
    for month in range(1, 12 + 1):
        (bigrams_in_cluster, bigrams_not_in_cluster, bigrams_in_cluster_tomorrow, bigrams_all_articles,
         bigrams_within_source) = get_bigrams_for_year_and_month_by_clusters(db_path, dct,
                                                                             tfidf_model, year,
                                                                             month, group_by, threshold)

        # convert bigrams list to bigrams shares
        total_bigrams_all_articles = _convert_bigrams_to_shares(bigrams_all_articles)
        total_bigrams_in_cluster = _convert_bigrams_to_shares(bigrams_in_cluster)
        total_bigrams_not_in_cluster = _convert_bigrams_to_shares(bigrams_not_in_cluster)
        total_bigrams_in_cluster_tomorrow = _convert_bigrams_to_shares(bigrams_in_cluster_tomorrow)

        # get the shares of top bigrams
        print('get shares of top bigrams in all articles')
        top_bigrams_freq_all_articles = get_shares_of_top1000_bigrams(top_bigrams, bigrams_all_articles)
        print('get shares of top bigrams for in cluster')
        top_bigrams_freq_in_cluster = get_shares_of_top1000_bigrams(top_bigrams, bigrams_in_cluster)
        print('get shares of top bigrams for not in cluster')
        top_bigrams_freq_not_in_cluster = get_shares_of_top1000_bigrams(top_bigrams, bigrams_not_in_cluster)
        print('get shares of top bigrams for in cluster tomorrow')
        top_bigrams_freq_in_cluster_tomorrow = get_shares_of_top1000_bigrams(top_bigrams, bigrams_in_cluster_tomorrow)

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

    aggregate_share_all_articles.to_csv(path_or_buf='../results/shares_all_articles_{}.csv'.format(year))
    aggregate_share_in_cluster.to_csv(path_or_buf='../results/shares_in_cluster_{}.csv'.format(year))
    aggregate_share_not_in_cluster.to_csv(path_or_buf='../results/shares_not_in_cluster_{}.csv'.format(year))
    aggregate_share_in_cluster_tomorrow.to_csv(path_or_buf='../results/shares_in_cluster_tomorrow_{}.csv'.format(year))

    print('standardizing bigram count for all articles')
    aggregate_share_in_cluster = standardize_bigrams_count(aggregate_share_in_cluster)
    print('standardizing bigram count in cluster')
    aggregate_share_not_in_cluster = standardize_bigrams_count(aggregate_share_not_in_cluster)
    print('standardizing bigram count not_in cluster')
    aggregate_share_in_cluster_tomorrow = standardize_bigrams_count(aggregate_share_in_cluster_tomorrow)
    print('standardizing bigram count for in cluster tomorrow')
    aggregate_share_all_articles = standardize_bigrams_count(aggregate_share_all_articles)

    aggregate_share_all_articles.to_csv(path_or_buf='../results/std_shares_all_articles_{}.csv'.format(year))
    aggregate_share_in_cluster.to_csv(path_or_buf='../results/std_shares_in_cluster_{}.csv'.format(year))
    aggregate_share_not_in_cluster.to_csv(path_or_buf='../results/std_shares_not_in_cluster_{}.csv'.format(year))
    aggregate_share_in_cluster_tomorrow.to_csv(path_or_buf='../results/std_shares_in_cluster_tomorrow_{}.csv'
                                               .format(year))

    print('calculating bias of news source by cluster groups')
    bias_all_articles, bias_in_cluster, bias_not_in_cluster, bias_in_cluster_tomorrow = parmap.map(
        calculate_bias,
        [
            aggregate_share_all_articles,
            aggregate_share_in_cluster,
            aggregate_share_not_in_cluster,
            aggregate_share_in_cluster_tomorrow],
        top1000_bigram)

    return _combine_bias_result_for_all_cluster(bias_all_articles, bias_in_cluster, bias_not_in_cluster,
                                                bias_in_cluster_tomorrow)


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

    if month is None:
        result = bias_averaged_over_year(db_path, dct, tfidf_model, top_1000_bigrams, year, group_by,
                                         threshold=threshold)
        result.to_csv(path_or_buf='../results/bias_{}_{}.csv'.format(year, group_by))
    else:
        result = bias_averaged_over_month(db_path, dct, tfidf_model, top_1000_bigrams, year, month, group_by,
                                          threshold=threshold)
        result.to_csv(path_or_buf='../results/bias_{}_{}_{}.csv'.format(year, month, group_by))


if __name__ == '__main__':
    main()
