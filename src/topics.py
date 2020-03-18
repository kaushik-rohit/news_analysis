import argparse
import db
from models import BoWIter1
from gensim import corpora, models
from gensim.corpora.mmcorpus import MmCorpus
from gensim.similarities import MatrixSimilarity
import bigrams
import helpers
import numpy as np
import os

# define necessary arguments to run the analysis
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

parser.add_argument('-t', '--threshold',
                    type=float,
                    default=0.1,
                    help='')


def build_dictionary_and_corpus(db_path, out):
    conn = db.NewsDb(db_path)
    print('fetching the top bigrams for all the topics')
    vocab = conn.select_top_bigrams(10000)
    print('building dictionary from top bigrams')
    topics_dct = corpora.Dictionary([vocab])
    print('saving the dictionary')
    topics_dct.save(os.path.join(out, 'topics_vocab.dict'))

    topics_corpus = []

    for _id in helpers.topics_id:
        print('building corpus for topic {}'.format(helpers.topics_id_to_name_map[_id]))
        topics_corpus.append(conn.get_corpus_for_topic(_id, vocab))

    bow_topic_corpus = []

    print(len(topics_corpus))
    print('building bag of words')
    for corpus in topics_corpus:
        bow_topic_corpus.append(topics_dct.doc2bow(corpus))

    print('transforming it to tfidf')
    tfidf = models.TfidfModel(iter(bow_topic_corpus))

    print('saving topics_corpus')
    corpora.MmCorpus.serialize(os.path.join(out, 'topics_corpus.mm'), iter(tfidf[bow_topic_corpus]))

    print('saving the tfidf model')
    tfidf.save(os.path.join(out, 'topics_tfidf'))

    conn.close()


def build_news_articles_corpus(articles, out, name):
    print('preprocessing articles and forming bigrams representation')
    bigram = bigrams.transform_corpus_to_bigrams(articles)
    dct = corpora.Dictionary.load('../models/topics_vocab.dict')
    print('creating bag of words representation')
    bow_corpus = []

    for bi in bigram:
        bow_corpus.append(dct.doc2bow(bi))
    print('transforming it to tfidf')
    tfidf = models.TfidfModel(iter(bow_corpus))

    print('saving the tfidf model')
    tfidf.save(os.path.join(out, 'topics_tfidf_{}'.format(name)))


def assign_topics_to_articles_for_month(db_path, year, month):
    conn = db.NewsDb(db_path)

    articles = list(conn.select_articles_by_year_and_month(year, month))
    topics = []

    print('loading models')
    topics_corpus = MmCorpus('../models/topics_corpus.mm')
    dct = corpora.Dictionary.load('../models/topics_vocab.dict')
    tfidf = models.TfidfModel.load('../models/topics_tfidf_2015_1')

    bigram_for_articles = bigrams.transform_corpus_to_bigrams(articles)
    bow_articles = BoWIter1(dct, bigram_for_articles)
    tfidf_articles = tfidf[bow_articles]

    index = MatrixSimilarity(topics_corpus, num_features=len(dct))
    for idx, similarities in enumerate(index[tfidf_articles]):
        max_sim_index = np.argmax(similarities)

        if not similarities[max_sim_index] >= 0.01:
            topics.append('None')
        else:
            topics.append(helpers.topics_id_to_name_map[helpers.topics_id[max_sim_index]])

    print('updating topics for all articles into db')
    conn.update_topic_for_articles(articles, topics)

    conn.close()


def main():
    args = parser.parse_args()
    db_path = args.db_path
    month = args.month
    year = args.year
    # conn = db.NewsDb(db_path)
    # articles = list(conn.select_articles_by_year_and_month(year, month))
    # build_news_articles_corpus(articles, '../models', '2015_1')
    # conn.close()
    # build_dictionary_and_corpus(db_path, '../models')
    assign_topics_to_articles_for_month(db_path, year, month)


if __name__ == '__main__':
    main()
