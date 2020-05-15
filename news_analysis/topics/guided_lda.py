import argparse
from gensim import corpora, models
from gensim.models.phrases import Phraser
from shared import db, helpers
from shared.models import BoWIter
import numpy as np

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

parser.add_argument('-p', '--phraser',
                    type=str,
                    required=True,
                    help='the path to gensim bigram/phraser model')

parser.add_argument('-tp', '--topics-prior',
                    type=str,
                    help='the path to expected topics bigrams')

parser.add_argument('-dict', '--dictionary',
                    type=str,
                    required=True,
                    help='the path to bag of words model')


def create_eta(priors, etadict, ntopics):
    eta = np.full(shape=(ntopics, len(etadict)), fill_value=1)  # create a (ntopics, nterms) matrix and fill with 1

    for word, topic in priors.items():  # for each word in the list of priors
        keyindex = [index for index, term in etadict.items() if term == word]  # look up the word in the dictionary

        if len(keyindex) > 0:  # if it's in the dictionary
            eta[topic, keyindex[0]] = 1e7  # put a large number in there
    eta = np.divide(eta, eta.sum(axis=0))  # normalize so that the probabilities sum to 1 over all topics
    return eta


def assign_topics_to_articles(articles, dictionary, phraser_model, priors):
    filter_fn = helpers.preprocess_text_for_lda

    print('converting corpus into bag of words')
    bow_articles = list(iter(BoWIter(dictionary, articles, filter_fn, phraser_model)))
    print('training lda')
    eta = create_eta(priors, dictionary, 20)
    lda_model = models.ldamulticore.LdaMulticore(corpus=bow_articles,
                                                 id2word=dictionary,
                                                 eta=eta,
                                                 num_topics=20)
    print(lda_model.print_topics())


def topics_month_helper(db_path, dct, phraser, year, month, priors):
    conn = db.NewsDb(db_path)
    articles = list(conn.select_articles_by_year_and_month(year, month))
    assign_topics_to_articles(articles, dct, phraser, priors)
    conn.close()


def topics_year_helper():
    pass


def main():
    args = parser.parse_args()
    phraser = Phraser.load(args.phraser)
    dct = corpora.Dictionary.load(args.dictionary)
    month = args.month
    year = args.year
    db_path = args.db_path
    topic_prior = helpers.load_json(args.topics_prior)

    topics_month_helper(db_path, dct, phraser, year, month, topic_prior)


if __name__ == '__main__':
    main()
