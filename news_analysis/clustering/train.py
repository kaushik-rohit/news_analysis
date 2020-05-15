import argparse
import os

from gensim import corpora
from gensim import models

from shared import helpers, db
from shared.models import BoWIter, CorpusIter

# create necessary arguments to run the analysis
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--db-path',
                    type=str,
                    required=True,
                    help='the path to database where news articles are stored')

parser.add_argument('-o', '--output-path',
                    type=str,
                    required=True,
                    help='output path where trained model should be stored')

parser.add_argument('-m', '--month',
                    choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    type=int,
                    default=None,
                    help='The month of data sources on which model is to be trained.'
                         'If month is not passed, model is trained on entire year.')

parser.add_argument('-y', '--year',
                    type=int,
                    default=2014,
                    help='The year of data sources on which model is to be trained')

parser.add_argument('-a', '--algo',
                    choices=['tfidf', 'bigram'],
                    type=str,
                    default='tfidf',
                    help='the algorithm to train news data on')


def train_tfidf(docs, out, name):
    """
    Parameters
    ----------
    @docs: (list) A list of strings, corpus on which TfIDF model will be trained
    @out: (string) directory path where bag of words and tfidf model will be saved
    @name: (string) suffix to be added to the models name, i.e if 2014 is passed as
        name arg, the saved models will be named tfidf_2014 and vocab_2014.dict

    Returns
    -------
    None
    """
    # build bag of words
    print('Building vocabulary for the corpus')
    corpus = CorpusIter(docs, helpers.preprocess_text)
    dictionary = corpora.Dictionary(iter(corpus))

    print('Bag of Words Done!! Trying feature reduction')
    print('length of dict before filtering is {}'.format(len(dictionary)))
    # downscale/feature reduction for dictionary
    # with a large dataset dictionary size can grow huge
    # leading to large tfidf and hence high runtime for transform to bow
    # and calculating similarities
    dictionary.filter_extremes(no_above=0.90)

    print('length of dict after filtering is {} ...'.format(len(dictionary)))
    print('creating bag of words for corpus and building tfidf model')
    # create tfidf
    bow_corpus = BoWIter(dictionary, docs)
    tfidf = models.TfidfModel(iter(bow_corpus))

    print('Saving trained models')
    # save the models for later use
    dictionary.save(os.path.join(out, 'vocab_{}.dict'.format(name)))
    # corpora.MmCorpus.serialize(os.path.join(out, 'bow_corpus_{}.mm'.format(name)), iter(BoWIter(dictionary, docs)))
    tfidf.save(os.path.join(out, 'tfidf_{}'.format(name)))


def main():
    args = parser.parse_args()

    conn = db.NewsDb(args.db_path)

    if args.month is None:
        n = conn.get_count_of_articles_for_year(args.year)
        corpus = list(conn.select_articles_by_year(args.year))
        assert(n == len(corpus))
        train_tfidf(corpus, args.output_path, '{}'.format(args.year))
    else:
        n = conn.get_count_of_articles_for_year_and_month(args.year, args.month)
        corpus = list(conn.select_articles_by_year_and_month(args.year, args.month))
        assert(n == len(corpus))
        train_tfidf(corpus, args.output_path, '{}_{}'.format(args.year, args.month))


if __name__ == '__main__':
    main()
