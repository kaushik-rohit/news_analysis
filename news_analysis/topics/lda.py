import argparse
from gensim import corpora, models
from gensim.models.phrases import Phraser
from shared import db, helpers
from shared.models import BoWIter

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
                    help='The years for which analysis is to be performed.')

parser.add_argument('-dict', '--dictionary',
                    type=str,
                    required=True,
                    help='the path to bag of words model')

parser.add_argument('-p', '--phraser-model',
                    type=str,
                    required=True,
                    help='the path to bigram model')

parser.add_argument('-l', '--lda-model',
                    type=str,
                    required=True,
                    help='the path to lda model that is to be used for topic labelling')


def assign_topics_to_articles(articles, dictionary, bigram_model, lda_model):
    filter_fn = helpers.preprocess_text_for_lda
    print('converting corpus into bag of words')
    bow_articles = list(iter(BoWIter(dictionary, articles, filter_fn, bigram_model)))
    print('fetching topics')
    topics = lda_model[bow_articles]
    print(topics[0], articles[0])
    return topics


def topics_month_helper(db_path, dct, bigram_model, lda_model, year, month):
    conn = db.NewsDb(db_path)
    articles = list(conn.select_articles_by_year_and_month(year, month))
    assign_topics_to_articles(articles, dct, bigram_model, lda_model)
    conn.close()


def topics_year_helper():
    pass


def main():
    args = parser.parse_args()
    dct = corpora.Dictionary.load(args.dictionary)
    lda_model = models.ldamulticore.LdaMulticore.load(args.lda_model)
    bigram_model = Phraser.load(args.phraser_model)
    month = args.month
    year = args.year
    db_path = args.db_path

    topics_month_helper(db_path, dct, bigram_model, lda_model, year, month)


if __name__ == '__main__':
    main()
