import argparse
from datetime import date, timedelta
import calendar
from shared.models import CorpusIter, BoWIter
from gensim import corpora, models
from gensim.corpora.mmcorpus import MmCorpus
from gensim.similarities import MatrixSimilarity
from gensim.models.phrases import Phrases, Phraser
from clustering import bigrams
from shared import helpers, db
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

parser.add_argument('-algo', '--algorithm',
                    type=str,
                    choices=['lda', 'tfidf', 'word2vec'],
                    default='tfidf',
                    help='which training algorithm to use')

parser.add_argument('-t', '--threshold',
                    type=float,
                    default=0.1,
                    help='')






class LDA:
    def __init__(self, db_path, models_path):
        self.db_path = db_path
        self.models_path = models_path

    @staticmethod
    def get_bigram_model_lda(articles):
        preprocess_fn = helpers.preprocess_text_for_lda
        corpus = CorpusIter(articles, preprocess_fn)
        phrases = Phrases(corpus, min_count=1, threshold=1)
        bigram = Phraser(phrases)
        return bigram

    @staticmethod
    def assign_topics_to_articles_for_month(year, month):
        conn = db.NewsDb(db_path)

        print('getting articles')
        preprocess_fn = helpers.preprocess_text_for_lda
        articles = list(conn.select_articles_by_year_and_month(year, month))
        corpus = CorpusIter(articles, preprocess_fn)

        print('get bigram model')
        bigram = get_bigram_model_lda(articles)
        bigram_corpus = bigram[iter(corpus)]

        print('building dictionary')
        dictionary = corpora.Dictionary(bigram_corpus)
        dictionary.filter_extremes(no_below=10, no_above=0.4)
        print('size of vocab: {}'.format(len(dictionary)))

        bow_articles = list(iter(BoWIter(dictionary, articles, preprocess_fn, bigram)))

        print('training lda')
        lda_model = models.ldamulticore.LdaMulticore(corpus=bow_articles,
                                                     id2word=dictionary,
                                                     num_topics=20)

        print(lda_model.print_topics())

        print(lda_model[bow_articles[0]])
        lda_model.save('../models/lda_{}_{}'.format(year, month))
        conn.close()


class TFIDF:
    def __init__(self, db_path):
        self.db_path = db_path

    @staticmethod
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

    def build_dictionary_and_corpus(self, out):
        conn = db.NewsDb(self.db_path)
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

    def assign_topics_to_articles_for_date_tfidf(self, cur_date):
        conn = db.NewsDb(self.db_path)
        articles = list(conn.select_articles_by_date(cur_date))

        print('assign topics to articles for {}'.format(cur_date))

        topics_corpus = MmCorpus('../models/topics_corpus.mm')
        dct = corpora.Dictionary.load('../models/topics_vocab.dict')

        bigram_for_articles = bigrams.transform_corpus_to_bigrams(articles)
        bow_articles = BoWIter(dct, bigram_for_articles)
        tfidf = models.TfidfModel(iter(bow_articles))
        tfidf_articles = tfidf[bow_articles]
        topics = []

        index = MatrixSimilarity(topics_corpus, num_features=len(dct))
        for idx, similarities in enumerate(index[tfidf_articles]):
            max_sim_index = np.argmax(similarities)

            if not similarities[max_sim_index] >= 0.01:
                topics.append('None')
            else:
                topics.append(helpers.topics_id_to_name_map[helpers.topics_id[max_sim_index]])

        conn.update_topic_for_articles(articles, topics)

        conn.close()

    @staticmethod
    def assign_topics_to_articles_for_month(year, month):
        start_date = date(year, month, 1)
        end_date = date(year, month, calendar.monthrange(year, month)[1])  # calendar.monthrange(year, month)[1]

        date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
        print('assign topics to articles for {} {}'.format(year, calendar.month_name[month]))
        for curr_date in date_range:
            assign_topics_to_articles_for_date_tfidf(curr_date)


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
    # assign_topics_to_articles_for_month_tfidf(db_path, year, month)
    assign_topics_to_articles_for_month_lda(db_path, year, month)


if __name__ == '__main__':
    main()
