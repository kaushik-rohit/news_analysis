from semantic_text_similarity.models import WebBertSimilarity
from tqdm import tqdm
import pandas as pd
import argparse
import nltk
import os


# create necessary arguments to run the analysis
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dir',
                    type=str,
                    required=True,
                    help='the path to directory where bbc transcripts are stored')

parser.add_argument('-y', '--year',
                    choices=[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
                             2015, 2016, 2017, 2018],
                    type=int,
                    required=True,
                    help='The years for which analysis is to be performed.')

parser.add_argument('-i', '--id',
                    type=int,
                    required=True,
                    help='id of bbc source for which transcripts are partitioned')

parser.add_argument('-a', '--algo',
                    type=str,
                    choices=[''],
                    help='algorithm to use for partitioning')


def load_data(path, bbc_id, year):
    path = os.path.join(path, '{}/{}/transcripts'.format(bbc_id, year))
    files = os.listdir(path)

    data = []
    for source in files:
        data.append(pd.read_csv(os.path.join(path, source)))
    df = pd.concat(data)
    df = df.drop(['Unnamed: 0', 'Has Transcript', 'Unavailable link', 'Unavailable reason'], axis=1)
    df = df.reset_index(drop=True)
    return df


def filter_short_sentence(sentences, max_len=9):
    """
    :param max_len: (int) length below which strings are dropped
    :param sentences: (list of strings)
    :return: (list of strings)
        list with all the string less than length l removed
    """
    n_sent = len(sentences)
    filtered_sentences = []
    lengths = []
    for i in range(n_sent):
        _len = len(sentences[i].split())
        lengths.append(_len)
        if _len >= max_len:
            filtered_sentences.append(sentences[i])
    return filtered_sentences


def partition_transcript_into_topics(transcript):
    web_model = WebBertSimilarity(device='cpu', batch_size=10)  # defaults to GPU prediction
    cluster = []
    sentences = nltk.sent_tokenize(transcript)
    sentences = filter_short_sentence(sentences)
    if len(sentences) == 0:
        return ''
    n_sent = len(sentences)
    current_cluster = [sentences[0]]
    n_cluster = 1

    i = 1
    while i <= n_sent:
        flag = False
        n_front = 0
        while (i + n_front) < n_sent and n_front <= 2:
            j = len(current_cluster) - 1
            n_back = 0
            while j >= 0 and n_back <= 2:
                sim = web_model.predict([(sentences[i + n_front], current_cluster[j])])[0] / 5
                #                 print(sentences[i+n_front])
                #                 print(current_cluster[j])
                #                 print(sim)
                #                 print('===============================')
                j = j - 1
                n_back = n_back + 1
                if sim >= 0.12:
                    current_cluster.extend(sentences[i:i + n_front + 1])
                    if i >= n_sent - 1:
                        cluster.append(' '.join(current_cluster))
                    flag = True
                    break

            if flag:
                break
            n_front = n_front + 1
        if not flag and i < n_sent:
            n_cluster += 1
            cluster.append(' '.join(current_cluster))
            current_cluster = [sentences[i]]
            i = i + 1
        else:
            i = i + n_front + 1

    return '\n---------------------\n'.join(cluster)


def partition_transcript_into_topics_old(transcript):
    web_model = WebBertSimilarity(device='cpu', batch_size=10)  # defaults to GPU prediction
    cluster = []
    sentences = nltk.sent_tokenize(transcript)
    sentences = filter_short_sentence(sentences)
    n_sent = len(sentences)
    current_cluster = [sentences[0]]
    n_cluster = 1
    for i in range(1, n_sent):
        sim = web_model.predict([(sentences[i], sentences[i - 1])])[0] / 5

        #         for sent in current_cluster:
        #             sim+= web_model.predict([(sent, sentences[i])])[0]/5

        #         avg_sim = sim/len(current_cluster)

        if sim >= 0.1:
            current_cluster.append(sentences[i])
            if i == n_sent - 1:
                cluster.append(' '.join(current_cluster))
        else:
            n_cluster += 1
            cluster.append(' '.join(current_cluster))
            current_cluster = [sentences[i]]
    return '\n---------------------\n'.join(cluster)


def partition_bbc_transcripts(path, out_path, bbc_id, year, algo='old'):
    """
    :param out_path: (string) path to directory where the partitioned transcripts are stored
    :param path: (string) path to directory where bbc transcripts are stored
    :param bbc_id: (int) id of bbc source
    :param year: (int) year of transcript that is to be partitioned
    :return:
    """
    df = load_data(path, bbc_id, year)
    transcripts = df['Transcript'].values

    partitions = []
    for transcript in tqdm(transcripts):
        partitions.append(partition_transcript_into_topics_old(transcript))

    df['partitioned_transcript'] = partitions
    df.to_csv('./bert_partitions_{}_{}.csv'.format(bbc_id, year))


if __name__ == '__main__':
    pass
