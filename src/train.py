import argparse
from gensim import models
from gensim import corpora
from dateutil import parser as date_parser
from sqlite3 import Error
from gensim.parsing.preprocessing import preprocess_string
import db
import os
from models import BoWIter, CorpusIter

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
                    default=None)

parser.add_argument('-y', '--year',
                    type=int,
                    default=2014)


def train_tfidf(docs, n, out, name):
    assert(n == len(docs))
    # build bag of words
    print('Building vocabulary for the corpus')
    corpus = CorpusIter(docs)
    dictionary = corpora.Dictionary(iter(corpus))

    print('length of dict before filtering is {}'.format(len(dictionary)))
    print('Bag of Words Done!! Trying feature reduction')
    # downscale/feature reduction for dictionary
    # with a large dataset dictionary size can grow huge
    # leading to large tfidf and hence high runtime for transform to bow
    # and calculating similarities
    dictionary.filter_extremes(no_above=0.99)

    print('length of dict after filtering is {} ...'.format(len(dictionary)))
    print('creating bag of words for corpus and building tfidf model')
    # create tfidf
    bow_corpus = BoWIter(dictionary, docs)

    tfidf = models.TfidfModel(iter(bow_corpus))

    print('Saving trained models')
    # save the models for later use
    dictionary.save(os.path.join(out, 'vocab_{}.dict'.format(name)))
    corpora.MmCorpus.serialize(os.path.join(out, 'bow_corpus_{}.mm'.format(name)), iter(BoWIter(dictionary, docs)))
    tfidf.save(os.path.join(out, 'tfidf_{}'.format(name)))


def main():
    args = parser.parse_args()

    conn = db.ArticlesDb(args.db_path)
    if args.month is None:
        n = conn.get_count_of_articles_for_year(args.year)
        corpus = conn.select_articles_by_year(args.year)
        train_tfidf(list(corpus), n, args.output_path, '{}'.format(args.year))
    else:
        n = conn.get_count_of_articles_for_year_and_month(args.year, args.month)
        corpus = conn.select_articles_by_year_and_month(args.year, args.month)
        train_tfidf(list(corpus), n, args.output_path, '{}_{}'.format(args.year, args.month))


if __name__ == '__main__':
    main()
