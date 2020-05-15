import argparse
import os
from gensim import corpora
from gensim.models.phrases import Phrases, Phraser
from shared import helpers
from shared import db
from shared.models import CorpusIter

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--db-path',
                    type=str,
                    required=True,
                    help='the path to database where news articles are stored')

parser.add_argument('-o', '--output-path',
                    type=str,
                    required=True,
                    help='output path where trained model should be stored')

parser.add_argument('-y', '--year',
                    choices=[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
                             2015, 2016, 2017, 2018],
                    type=int,
                    default=2014,
                    help='The year of data sources on which model is to be trained')


def build_phrase(docs, out, name):
    corpus = CorpusIter(docs, helpers.preprocess_text_for_lda)
    phrases = Phrases(iter(corpus), min_count=1, threshold=0.5, scoring='npmi')
    print(phrases.vocab)
    print('bigram model built!!')
    print('now computing the dictionary')
    dictionary = corpora.Dictionary(phrases[iter(corpus)])
    dictionary.filter_extremes(no_above=0.50)
    dictionary.save(os.path.join(out, 'vocab_{}.dict'.format(name)))

    bigram = Phraser(phrases)
    bigram.save(os.path.join(out, "bigram_{}.pkl".format(name)))


def main():
    args = parser.parse_args()
    conn = db.NewsDb(args.db_path)

    n = conn.get_count_of_articles_for_year_and_month(args.year, 1)
    corpus = list(conn.select_articles_by_year_and_month(args.year, 1))
    assert (n == len(corpus))
    build_phrase(corpus, args.output_path, '{}'.format(args.year))


if __name__ == '__main__':
    main()
