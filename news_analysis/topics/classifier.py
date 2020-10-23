import argparse

from gensim import corpora, models
from gensim.models import Doc2Vec
from gensim.models.phrases import Phraser
from shared import db, helpers
from joblib import dump, load
import topics.net as net
from shared.models import BoWIter, CorpusIter
import parmap
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

parser.add_argument('--model-name',
                    type=str,
                    choices=['lda', 'doc2vec'],
                    default='doc2vec',
                    help='what model to use for topic classification')

parser.add_argument('-dict', '--dictionary',
                    type=str,
                    required=False,
                    help='the path to bag of words model')

parser.add_argument('-p', '--phraser-model',
                    type=str,
                    required=False,
                    help='the path to bigram model')

parser.add_argument('-l', '--lda-model',
                    type=str,
                    required=False,
                    help='the path to lda model that is to be used for topic labelling')

parser.add_argument('--doc2vec',
                    type=str,
                    required=False,
                    help='the path to doc2vec model.')

parser.add_argument('--classifier',
                    type=str,
                    required=True,
                    help='the path to tensorflow nn classifier')

parser.add_argument('--encoder',
                    type=str,
                    required=False,
                    help='path to encoder for y labels in classification model')


class GenericClassifier:

    def __init__(self, db_path):
        self.db_path = db_path

    def classify_articles(self, articles):
        raise NotImplementedError("This class is not meant to be used.")

    def classify_articles_for_month(self, year, month):
        conn = db.NewsDb(self.db_path)
        articles = list(conn.select_articles_by_year_and_month(year, month))
        self.classify_articles(articles)
        conn.close()

    def classify_articles_for_year(self, year):
        for month in range(1, 13):
            self.classify_articles_for_month(year, month)


class LDAClassifier(GenericClassifier):
    def __init__(self, db_path, dictionary, bigram_model, lda_model, classifier):
        super(LDAClassifier, self).__init__(db_path)
        self.dict = dictionary
        self.phraser = bigram_model
        self.lda_model = lda_model
        self.classifier = classifier

    def classify_articles(self, articles):
        filter_fn = helpers.preprocess_text_for_lda
        print('converting corpus into bag of words')
        bow_articles = list(iter(BoWIter(self.dictionary, articles, filter_fn, self.phraser)))
        print('fetching topics')
        topics = self.lda_model[bow_articles]
        print(topics[0], articles[0])
        return topics


def get_top_n(prediction, topn=3):
    pred = [(i, p) for i, p in enumerate(prediction)]
    pred_sorted = sorted(pred, key=lambda x: x[1], reverse=True)
    top_pred = pred_sorted[:topn]
    return top_pred


def classify_articles_doc2vec(db_path, doc2vec, classifier, year, month):
    filter_fn = helpers.preprocess_text_for_doc2vec
    conn = db.NewsDb(db_path)
    print('getting articles from the database...')
    articles = list(conn.select_articles_by_year_and_month(year, month))
    corpus = list(iter(CorpusIter(articles, preprocess_fn=filter_fn)))
    print('inferring vector for news transcripts')
    corpus_vector = parmap.map(doc2vec.infer_vector, corpus, pm_pbar=True)
    corpus_vector = np.array(corpus_vector)
    corpus_vector = corpus_vector.reshape(len(corpus_vector), corpus_vector[0].shape[0])
    print('hurray!!performing predictions...')
    predictions = classifier.predict(corpus_vector)
    print('fetching top-3 predictions and storing it in db')
    top_predictions = parmap.map(get_top_n, predictions, pm_pbar=True)
    # print(top_predictions[0])
    # update the predictions to database
    conn.update_topic_for_articles(articles, top_predictions)
    conn.close()


def classify_articles_doc2vec_for_year(db_path, doc2vec, classifier, year):
    for month in range(1, 13):
        print('classifying articles for year {} and month {}'.format(year, month))
        classify_articles_doc2vec(db_path, doc2vec, classifier, year, month)


def main():
    args = parser.parse_args()
    name = args.model_name
    month = args.month
    year = args.year
    db_path = args.db_path

    if name == 'lda':
        dct = corpora.Dictionary.load(args.dictionary)
        lda_model = models.ldamulticore.LdaMulticore.load(args.lda_model)
        bigram_model = Phraser.load(args.phraser_model)
    elif name == 'doc2vec':
        doc2vec = Doc2Vec.load(args.doc2vec)
        encoder = load(args.encoder)
        classifier = net.doc2vec_network(encoder)
        classifier.load_weights(args.classifier)
        if month is None:
            classify_articles_doc2vec_for_year(db_path, doc2vec, classifier, year)
        else:
            classify_articles_doc2vec(db_path, doc2vec, classifier, year, month)


if __name__ == '__main__':
    main()
