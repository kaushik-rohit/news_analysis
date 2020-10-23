import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import shared
import nltk
import os


def count_of_bigram_shares_change(path):
    shares = pd.read_csv(os.path.join(path, 'final_dataframe_party_std_before.csv'))
    bigram_topics = pd.read_csv(os.path.join(path, 'bigram_topics.csv'))
    topics = bigram_topics['topic']
    bigram_topic_dict = {}

    for index, row in bigram_topics.iterrows():
        bigram_topic_dict[row['bigram']] = row['topic']

    res_df = pd.DataFrame(shares.bigrams.values, columns=['bigram'])
    res_df['jan-2016'] = shares['BBC+News jan-2016 54.csv']
    res_df['feb-2016'] = shares['BBC+News feb-2016 54.csv']
    res_df['diff'] = res_df['feb-2016'] - res_df['jan-2016']
    res_df['change'] = res_df['diff'].apply(lambda x: 1 if x > 0 else (0 if x == 0 else -1))
    res_df['topic'] = res_df['bigram'].apply(lambda x: bigram_topic_dict[x])

    res_df.to_csv(os.path.join(path, 'bigram_shares.csv'))
    count_df = res_df[['change', 'topic']]
    count_df['count'] = 1
    count_df = count_df.groupby(['change', 'topic']).sum()
    count_df.to_csv(os.path.join(path, 'bigram_share_change_count_by_topic.csv'))

    return res_df


def calculate_accuracy(preds):
    topics = preds.topic.unique()
    topics_count = preds.topic.value_counts()

    preds['predicted_topic'] = preds['predicted_topic'].apply(lambda x: ast.literal_eval(x))
    true_topic = preds['topic'].values
    pred_topic = preds['predicted_topic'].values
    total = len(true_topic)
    correct_pred_top1 = 0
    correct_pred_top3 = 0

    accuracy_by_topic = {topic: [0, 0] for topic in topics}

    for i in range(total):
        if true_topic[i] == pred_topic[i][0][0]:
            correct_pred_top1 += 1
            correct_pred_top3 += 1
            accuracy_by_topic[true_topic[i]][0] += 1
            accuracy_by_topic[true_topic[i]][1] += 1
        elif true_topic[i] == pred_topic[i][1][0]:
            correct_pred_top3 += 1
            accuracy_by_topic[true_topic[i]][1] += 1
        elif true_topic[i] == pred_topic[i][2][0]:
            correct_pred_top3 += 1
            accuracy_by_topic[true_topic[i]][1] += 1

    rows = []
    for key, value in accuracy_by_topic.items():
        rows += [[key, value[0]/topics_count[key], value[1]/topics_count[key]]]

    df = pd.DataFrame(rows, columns=['topic', 'top1-accuracy', 'top3-accuracy'])
    print(df)
    print('top1-accuracy: {}, top3-accuracy: {}'.format(correct_pred_top1/total, correct_pred_top3/total))
