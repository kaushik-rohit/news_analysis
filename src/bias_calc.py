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
import helpers
import multiprocessing as mp
from collections import Counter
from tqdm import tqdm
import numpy as np
import db
import string

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

porter = PorterStemmer()


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
    return list(filter(lambda token: token not in string.punctuation and token not in stop_words and
                (len(token) > 1), sent))


def get_cluster_for_the_day(path, dct, tfidf_model, curr_date, threshold):
    delta = timedelta(days=1)
    next_date = curr_date + delta
    conn = db.ArticlesDb(path)

    print('calculating cluster groups for {}'.format(curr_date))
    articles_day1 = list(conn.select_articles_by_date(curr_date))
    articles_day2 = list(conn.select_articles_by_date(next_date))

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
    @year
    @month
    @threshold

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

    pool = mp.Pool(mp.cpu_count())  # calculate stats for date in parallel
    print('Parallelize on {} CPUs'.format(mp.cpu_count()))
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    stats = pool.starmap(get_cluster_for_the_day, [(path, dct, tfidf_model, curr_date, threshold)
                                                   for curr_date in date_range])
    pool.close()

    for stat in stats:
        in_cluster_articles += stat[0]
        not_in_cluster_articles += stat[1]
        not_in_cluster_but_next_day_cluster += stat[2]

    return in_cluster_articles, not_in_cluster_articles, not_in_cluster_but_next_day_cluster


def get_bigrams_for_single_article(article):
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

    return article.source, bigram


def get_bigrams_in_articles(articles):
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

    all_bigrams = {}
    bigrams_by_source = []

    pool = mp.Pool(mp.cpu_count())  # using parallel processing

    for _ in tqdm(pool.imap_unordered(get_bigrams_for_single_article, articles), total=len(articles)):
        bigrams_by_source.append(_)

    pool.close()

    for source, bigrams in bigrams_by_source:

        if source in all_bigrams:
            all_bigrams[source] += bigrams
        else:
            all_bigrams[source] = bigrams

    for source, bigrams in all_bigrams.items():
        bigrams_freq = Counter(bigrams)
        total = sum(bigrams_freq.values(), 0.0)

        for bigram in bigrams_freq:
            bigrams_freq[bigram] /= total

        all_bigrams[source] = bigrams_freq

    return all_bigrams


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


def get_freq_of_top1000_bigrams(top1000_bigram, bigrams):
    """
    Calculates the frequency of top 1000 bigrams occurring in bigrams grouped by news source
    Parameters
    ----------
    @top1000_bigram: (list) of 1000 bigrams from MP speeches
    @bigrams: (dictionary) of bigrams calculated from news source with key as source

    Returns
    -------
    a pandas DataFrame with count of each top 1000 bigrams in different news source
    """
    rows = []

    assert (len(top1000_bigram) == 1000)

    for source, bigram_freq in tqdm(bigrams.items()):
        top1000_bigram_freq = [0] * 1000

        for i in range(1000):
            if top1000_bigram[i] in bigram_freq:
                top1000_bigram_freq[i] = bigram_freq[top1000_bigram[i]]
        rows += [[source] + top1000_bigram_freq]

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
            assert (isinstance(alpha, float))
            assert (isinstance(beta, float))
            # print('alpha: {}, beta: {}'.format(alpha, beta))
            if np.isnan(row[bigrams[i]]):
                continue
            num += beta * (row[bigrams[i]] - alpha)
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
    sources = list(all_articles.keys()) + list(in_cluster.keys()) + list(not_in_cluster.keys()) + \
              list(in_tomorrow_cluster.keys())

    sources = list(set(sources))
    rows = []
    for source in sources:
        bias_all_articles = all_articles[source] if source in all_articles else 0
        bias_in_cluster = in_cluster[source] if source in in_cluster else 0
        bias_not_in_cluster = not_in_cluster[source] if source in not_in_cluster else 0
        bias_in_tomorrow_cluster = in_tomorrow_cluster[source] if source in in_tomorrow_cluster else 0

        rows += [[source, bias_all_articles, bias_in_cluster, bias_not_in_cluster, bias_in_tomorrow_cluster]]

    return pd.DataFrame(rows, columns=columns).sort_values(by=['source']).reset_index(drop=True)


def aggregate_month_bias_results(results):
    """
    Parameters
    ----------
    @results: (list) of dictionary tuples with dictionary as monthly bias for each cluster

    Returns
    -------
    (a pandas dataframe) with bias averaged over month grouped by news source
    """

    bias_all_articles = {}
    bias_in_cluster = {}
    bias_not_in_cluster = {}
    bias_in_cluster_tomorrow = {}

    for result in results:
        all_articles, in_cluster, not_in_cluster, in_cluster_tomorrow = (result[0], result[1], result[2], result[3])

        for source, bias in all_articles.items():
            if source not in bias_all_articles:
                bias_all_articles[source] = [bias]
            else:
                bias_all_articles[source] += [bias]

        for source, bias in in_cluster.items():
            if source not in bias_in_cluster:
                bias_in_cluster[source] = [bias]
            else:
                bias_in_cluster[source] += [bias]

        for source, bias in not_in_cluster.items():
            if source not in bias_not_in_cluster:
                bias_not_in_cluster[source] = [bias]
            else:
                bias_not_in_cluster[source] += [bias]

        for source, bias in in_cluster_tomorrow.items():
            if source not in bias_in_cluster_tomorrow:
                bias_in_cluster_tomorrow[source] = [bias]
            else:
                bias_in_cluster_tomorrow[source] += [bias]

        sources = list(bias_all_articles.keys()) + list(bias_in_cluster.keys()) + list(bias_not_in_cluster.keys()) + \
                  list(bias_in_cluster_tomorrow.keys())

        sources = list(set(sources))

        for source in sources:
            bias_all_articles[source] = np.mean(bias_all_articles[source])
            bias_in_cluster[source] = np.mean(bias_in_cluster[source])
            bias_not_in_cluster[source] = np.mean(bias_not_in_cluster[source])
            bias_in_cluster_tomorrow[source] = np.mean(bias_in_cluster_tomorrow[source])

    return _combine_bias_result_for_all_cluster(bias_all_articles, bias_in_cluster, bias_not_in_cluster,
                                                bias_in_cluster_tomorrow)


def bias_averaged_over_month(db_path, dct, tfidf_model, top1000_bigram, year, month, agg_later=False, threshold=0.3):
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

    conn = db.ArticlesDb(db_path)
    news_source = conn.get_news_source_for_month(year, month)
    all_articles = list(conn.select_articles_by_year_and_month(year, month))
    n_articles = conn.get_count_of_articles_for_year_and_month(year, month)

    assert (n_articles == len(all_articles))

    top_bigrams = top1000_bigram['bigram'].tolist()
    in_cluster, not_in_cluster, in_cluster_tomorrow = get_cluster_of_articles(db_path, dct, tfidf_model,
                                                                              year, month, threshold)

    assert (n_articles == (len(in_cluster) + len(not_in_cluster)))

    print('calculating bigrams for in cluster articles')
    bigrams_in_cluster = get_bigrams_in_articles(in_cluster)
    print('calculating bigrams for not in cluster articles')
    bigrams_not_in_cluster = get_bigrams_in_articles(not_in_cluster)
    print('calculating bigrams for in tomorrow cluster articles')
    bigrams_in_cluster_tomorrow = get_bigrams_in_articles(in_cluster_tomorrow)
    print('calculating bigrams for all articles')
    bigrams_all_articles = get_bigrams_in_articles(all_articles)

    del all_articles, in_cluster, not_in_cluster, in_cluster_tomorrow

    pool = mp.Pool(mp.cpu_count())
    print('calculating frequency of top MPs bigrams in news articles')
    (top_bigrams_freq_all_articles, top_bigrams_freq_in_cluster, top_bigrams_freq_not_in_cluster,
     top_bigrams_freq_in_cluster_tomorrow) = pool.starmap(get_freq_of_top1000_bigrams,
                                                          [(top_bigrams, bigrams_all_articles),
                                                           (top_bigrams, bigrams_in_cluster),
                                                           (top_bigrams, bigrams_not_in_cluster),
                                                           (top_bigrams, bigrams_in_cluster_tomorrow)])
    pool.close()

    top_bigrams_freq_all_articles.to_csv(path_or_buf='../results/top_bigrams_freq_all_articles_{}_{}.csv'.format(year,
                                                                                                                 month))
    top_bigrams_freq_in_cluster.to_csv(path_or_buf='../results/top_bigrams_freq_in_cluster_{}_{}.csv'.format(year,
                                                                                                             month))
    top_bigrams_freq_not_in_cluster.to_csv(path_or_buf='../results/top_bigrams_freq_not_in_cluster_{}_{}.csv'.format(
        year, month))
    top_bigrams_freq_in_cluster_tomorrow.to_csv(path_or_buf='../results/top_bigrams_freq_in_cluster_tomorrow_{}_{}.csv'
                                                .format(year, month))

    del bigrams_in_cluster, bigrams_in_cluster_tomorrow, bigrams_not_in_cluster, bigrams_all_articles

    print('standardizing bigram count for all articles')
    standardize_bigrams_count(top_bigrams_freq_all_articles)
    print('standardizing bigram count in cluster')
    standardize_bigrams_count(top_bigrams_freq_in_cluster)
    print('standardizing bigram count not_in cluster')
    standardize_bigrams_count(top_bigrams_freq_not_in_cluster)
    print('standardizing bigram count for in cluster tomorrow')
    standardize_bigrams_count(top_bigrams_freq_in_cluster_tomorrow)

    top_bigrams_freq_all_articles.to_csv(path_or_buf='../results/top_bigrams_freq_all_articles_std_{}_{}.csv'.format(
        year, month))
    top_bigrams_freq_in_cluster.to_csv(path_or_buf='../results/top_bigrams_freq_in_cluster_std_{}_{}.csv'.format(year,
                                                                                                                 month))
    top_bigrams_freq_not_in_cluster.to_csv(path_or_buf='../results/top_bigrams_freq_not_in_cluster_std_{}_{}.csv'.
                                           format(year, month))
    top_bigrams_freq_in_cluster_tomorrow.to_csv(
        path_or_buf='../results/top_bigrams_freq_in_cluster_tomorrow_std_{}_{}.csv'.format(year, month))

    pool = mp.Pool(mp.cpu_count())  # calculate stats for date in parallel
    print('calculating bias of news source by cluster groups')
    bias_all_articles, bias_in_cluster, bias_not_in_cluster, bias_in_cluster_tomorrow = pool.starmap(
        calculate_bias, [(top_bigrams_freq_all_articles, top1000_bigram),
                         (top_bigrams_freq_in_cluster, top1000_bigram),
                         (top_bigrams_freq_not_in_cluster, top1000_bigram),
                         (top_bigrams_freq_in_cluster_tomorrow, top1000_bigram)])
    pool.close()

    print(bias_all_articles)

    if agg_later:
        return bias_all_articles, bias_in_cluster, bias_not_in_cluster, bias_in_cluster_tomorrow

    return _combine_bias_result_for_all_cluster(bias_all_articles, bias_in_cluster, bias_not_in_cluster,
                                                bias_in_cluster_tomorrow)


def bias_averaged_over_year(db_path, dct, tfidf_model, top1000_bigrams, year, threshold=0.3):
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
    a pandas dataframe with bias for different clusters grouped by souce
    """

    assert (1 > threshold > 0)
    results = []

    for month in range(1, 12 + 1):
        results += [bias_averaged_over_month(db_path, dct, tfidf_model, top1000_bigrams, year, month, agg_later=True,
                                             threshold=threshold)]

    assert (len(results) == 12)

    bias = aggregate_month_bias_results(results)

    return bias


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

    if month is None:
        result = bias_averaged_over_year(db_path, dct, tfidf_model, top_1000_bigrams, year, threshold=threshold)
        result.to_csv(path_or_buf='../results/bias_{}.csv'.format(year))
    else:
        result = bias_averaged_over_month(db_path, dct, tfidf_model, top_1000_bigrams, year, month, threshold=threshold)
        result.to_csv(path_or_buf='../results/bias_{}_{}.csv'.format(year, month))


if __name__ == '__main__':
    main()
