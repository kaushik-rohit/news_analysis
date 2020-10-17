import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import os


topics_id = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
             '19', '20', '21']

topics_id_to_name_map = {
    '02': 'Agriculture, animals, food and rural affairs',
    '03': 'Asylum, immigration and nationality',
    '04': 'Business, industry and consumers',
    '05': 'Communities and families',
    '06': 'Crime, civil law, justice and rights',
    '07': 'Culture, media and sport',
    '08': 'Defence',
    '09': 'Economy and finance',
    '10': 'Education',
    '11': 'Employment and training',
    '12': 'Energy and environment',
    '13': 'European Union',
    '14': 'Health services and medicine',
    '15': 'Housing and planning',
    '16': 'International affairs',
    '17': 'Parliament, government and politics',
    '18': 'Science and technology',
    '19': 'Social security and pensions',
    '20': 'Social services',
    '21': 'Transport'
}


def get_keywords_occurance_position(path_to_csv):
    """
    :param path_to_csv:
    :return:
    """

    keywords = ['control.border', 'control.immigr', '350.million']

    df = pd.read_csv(path_to_csv)

    def get_bigrams(transcript):
        stopW = stopwords.words('english')
        ps = PorterStemmer()
        # to lower case
        clean_transcript = transcript.lower()
        clean_transcript = word_tokenize(clean_transcript)
        # remove stopwords and single characters
        clean_transcript = [i for i in clean_transcript if i not in stopW and len(i) > 1]
        # stemming
        clean_transcript = [ps.stem(word) for word in clean_transcript]

        # bigrams
        phrases = list(nltk.bigrams(clean_transcript))
        phrases = [phrase[0] + '.' + phrase[1] for phrase in phrases]
        return phrases

    position_rep = []
    overall_positions = {keyword: [] for keyword in keywords}
    for index, row in df.iterrows():
        bigrams = get_bigrams(row['Transcript'])
        n_bigrams = len(bigrams)

        positions = {keyword: [] for keyword in keywords}

        for i, x in enumerate(bigrams):
            for keyword in keywords:
                if x == keyword:
                    positions[keyword].append(i)

        string_rep = ''

        for keyword in keywords:
            position_for_keyword = positions[keyword]
            position_for_keyword = [round(p / n_bigrams, 2) for p in position_for_keyword]
            overall_positions[keyword].append(position_for_keyword)

    i = 0
    for keyword in keywords:
        df['keyword' + str(i)] = keyword
        df['position' + str(i)] = overall_positions[keyword]
        i += 1

    return df


def assign_bigrams_to_topic(path):
    si = pd.read_csv(os.path.join(path, 'si_before.csv'))
    si_dict = {}
    for index, row in si.iterrows():
        si_dict[row['bigrams']] = row['si_before']

    res_df = pd.DataFrame(si['bigrams'].values, columns=['bigram'])
    topics_df = pd.DataFrame(si['bigrams'].values, columns=['bigram'])
    for topic in topics_id:
        sit = pd.read_csv(os.path.join(path, 'sit_before{}.csv'.format(topic)))
        score = []

        for index, row in sit.iterrows():
            bigram = row['bigrams']
            sit_before = row['sit_before']
            score.append(sit_before / si_dict[bigram])
        sit['sit/si'] = score
        res_df[topic] = score
        # sit = sit.sort_values(by=['sit/si'], ascending=False).reset_index(drop=True)
        sit.to_csv(os.path.join(path, 'sit_before{}.csv'.format(topic)))

    res_df.to_csv(os.path.join(path, 'bigram_sit_si_score.csv'))
    topics_df['topic'] = res_df[res_df.columns[1:]].idxmax(axis=1)
    topics_df['topic'] = topics_df['topic'].apply(lambda x: topics_id_to_name_map[x])
    topics_df.to_csv(os.path.join(path, 'bigram_topics.csv'))


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
    res_df['change'] = res_df['diff'].apply(lambda x: 1 if x > 0 else(0 if x == 0 else -1))
    res_df['topic'] = res_df['bigram'].apply(lambda x: bigram_topic_dict[x])

    res_df.to_csv(os.path.join(path, 'bigram_shares.csv'))
    count_df = res_df[['change', 'topic']]
    count_df['count'] = 1
    count_df = count_df.groupby(['change', 'topic']).sum()
    count_df.to_csv(os.path.join(path, 'bigram_share_change_count_by_topic.csv'))

    return res_df
