import itertools
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter
import string
import parmap
import helpers

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
    return list(filter(lambda token: token not in string.punctuation and token not in stop_words and (
            len(token) > 1), sent))


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


def get_bigrams_in_articles(articles, group_by, pbar=True):
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

    all_bigrams = {source: [] for source in helpers.source_names}

    bigrams_by_source = parmap.map(get_bigrams_for_single_article, articles, group_by, pm_pbar=pbar)

    for source, bigrams in bigrams_by_source:
        all_bigrams[source] += bigrams

    return all_bigrams


def convert_bigrams_to_shares(all_bigrams):
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


def convert_bigrams_to_shares_grouped_by_source(bigrams_by_source):
    total_count_by_source = {key: None for key in bigrams_by_source.keys()}

    for source, bigrams in bigrams_by_source.items():
        total_count_by_source[source] = convert_bigrams_to_shares(bigrams)

    return total_count_by_source


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

    for bigram in bigram_set:
        std = top_bigrams_share_by_source[bigram].std()
        mean = top_bigrams_share_by_source[bigram].mean()
        top_bigrams_share_by_source[bigram] = (top_bigrams_share_by_source[bigram] - mean) / std

    return top_bigrams_share_by_source.fillna(0)  # if z score is nan change it 0


def standardize_bigrams_count_group_by_source(top_bigrams_share_by_source):
    for source, bigram in top_bigrams_share_by_source.items():
        top_bigrams_share_by_source[source] = standardize_bigrams_count(top_bigrams_share_by_source[source])
