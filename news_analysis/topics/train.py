import argparse
import os
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
from shared import helpers
from shared import db
from shared.models import CorpusIter, BoWIter
import numpy as np
import matplotlib.pyplot as plt

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

parser.add_argument('-dict', '--dictionary',
                    type=str,
                    default=None,
                    help='the path to vocab. only required training lda')

parser.add_argument('-p', '--phraser',
                    type=str,
                    required=True,
                    help='the path to gensim bigram/phraser model')

parser.add_argument('-tp', '--topics-prior',
                    type=str,
                    help='the path to expected topics bigrams')

parser.add_argument('-t', '--type',
                    type=str,
                    choices=['dict', 'lda'],
                    required=True,
                    help='build the dictionary or train lda')


def build_phrase(docs, out, name):
    corpus = CorpusIter(docs, helpers.preprocess_text_for_lda)
    # phrases = Phrases(iter(corpus), min_count=1, threshold=0.5, scoring='npmi')
    # print(phrases.vocab)
    print('bigram model built!!')
    print('now computing the dictionary')
    dictionary = corpora.Dictionary(iter(corpus))
    dictionary.filter_extremes(no_above=0.40, no_below=10)
    dictionary.save(os.path.join(out, 'topics_vocab_{}.dict'.format(name)))

    # bigram = Phraser(phrases)
    # bigram.save(os.path.join(out, "bigram_{}.pkl".format(name)))


def create_eta(priors, etadict, ntopics):
    eta = np.full(shape=(ntopics, len(etadict)), fill_value=1)  # create a (ntopics, nterms) matrix and fill with 1
    for word, topic in priors.items():  # for each word in the list of priors
        keyindex = [index for index, term in etadict.items() if term == word]  # look up the word in the dictionary

        if len(keyindex) > 0:  # if it's in the dictionary
            eta[topic, keyindex[0]] = 1e7  # put a large number in there
    eta = np.divide(eta, eta.sum(axis=0))  # normalize so that the probabilities sum to 1 over all topics
    return eta


def viz_model(model, modeldict):
    ntopics = model.num_topics
    # top words associated with the resulting topics
    topics = ['Topic {}: {}'.format(t, modeldict[w]) for t in range(ntopics) for w,p in model.get_topic_terms(t, topn=1)]
    terms = [modeldict[w] for w in modeldict.keys()]
    fig, ax = plt.subplots()
    ax.imshow(model.get_topics())  # plot the numpy matrix
    ax.set_xticks(modeldict.keys())  # set up the x-axis
    ax.set_xticklabels(terms, rotation=90)
    ax.set_yticks(np.arange(ntopics))  # set up the y-axis
    ax.set_yticklabels(topics)
    plt.savefig('./lda_results.png')


def train_lda(docs, priors, dictionary, out, name):
    filter_fn = helpers.preprocess_text_for_lda

    print('converting corpus into bag of words')
    bow_articles = list(iter(BoWIter(dictionary, docs, filter_fn)))
    print('training lda')
    eta = create_eta(priors, dictionary, 20)
    lda_model = models.ldamulticore.LdaMulticore(corpus=bow_articles,
                                                 id2word=dictionary,
                                                 eta=eta,
                                                 random_state=42,
                                                 per_word_topics=True,
                                                 iterations=100,
                                                 num_topics=20)

    lda_model.save(os.path.join(out, 'lda_model_{}.pkl'.format(name)))
    print(lda_model.print_topics())


def main():
    args = parser.parse_args()
    conn = db.NewsDb(args.db_path)

    if args.type == 'dict' and args.dictionary is not None:
        print('error, dictionary passed with type dict!')
        return

    train = args.type
    n = conn.get_count_of_articles_for_year_and_month(args.year, 1)
    corpus = list(conn.select_articles_by_year_and_month(args.year, 1))
    assert (n == len(corpus))
    if train == 'dict':
        build_phrase(corpus, args.output_path, '{}'.format(args.year))
    elif train == 'lda':
        priors = helpers.load_json(args.topics_prior)
        dct = corpora.Dictionary.load(args.dictionary)
        train_lda(corpus, priors, dct, args.output_path, '{}'.format(args.year))


if __name__ == '__main__':
    main()
