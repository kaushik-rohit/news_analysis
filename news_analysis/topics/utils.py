from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from shared import helpers, db
import pandas as pd
import numpy as np
import sqlite3
import parmap
import nltk
import ast
import os
import re


def get_category(topic, mapp):
    category = mapp.loc[mapp['Topic'] == topic]
    return category['Category'].iloc[0]


def preprocess_speech_data(df, path, name, mapp):
    # df = df.drop(['speaker'], axis=1)
    df['title'] = df.apply(lambda x: x['title'].strip(), axis=1)
    df['topic'] = df.apply(lambda x: get_category(x['title'], mapp), axis=1)
    df.to_csv(os.path.join(path, name), index=False)


def count_by_party(row, party):
    row_party = row['Party']

    if party == 'Others' and row_party not in ['Labour', 'Conservative']:
        return len(row['Speech'].split())
    if row['Party'] == party:
        return len(row['Speech'].split())
    return 0


def group_speech_into_debates(path, name, mapp):
    df = pd.read_csv(os.path.join(path, name), index_col=0)
    df = df.groupby(['date', 'topic'])['transcript'].apply(lambda x: ' '.join(x)).reset_index()
    preprocess_speech_data(df, path, name.replace('speech', 'debate'), mapp)


def group_speech_into_debates_with_govt_count(path, name, mapp):
    df = pd.read_csv(os.path.join(path, name))
    df['Labour'] = df.apply(lambda x: count_by_party(x, 'Labour'), axis=1)
    df['Conservative'] = df.apply(lambda x: count_by_party(x, 'Conservative'), axis=1)
    df['Others'] = df.apply(lambda x: count_by_party(x, 'Others'), axis=1)
    df = df.groupby(['Date', 'Topic'], as_index=False).agg({'Speech': lambda x: ' '.join(x),
                                                            'Labour': 'sum',
                                                            'Conservative': 'sum',
                                                            'Others': 'sum'})
    df.columns = ['date', 'title', 'transcript', 'labour', 'conservative', 'others']
    preprocess_speech_data(df, path, name.replace('commons_with_govt', 'debate'), mapp)


def save_articles_to_csv(year):
    conn = sqlite3.connect('../data/news.db')
    sql = 'select * from articles where year={}'.format(year)
    df = pd.read_sql(sql, conn)
    print(df.head())
    df['top1_topic'] = df.apply(lambda row: helpers.topics_index_to_name_map[row['top1_topic']], axis=1)
    df['top2_topic'] = df.apply(lambda row: helpers.topics_index_to_name_map[row['top2_topic']], axis=1)
    df['top3_topic'] = df.apply(lambda row: helpers.topics_index_to_name_map[row['top3_topic']], axis=1)
    df.to_csv('news_{}_predictions.csv'.format(year))


def is_political(article, political_bigrams):
    transcript = article.transcript
    stopW = nltk.corpus.stopwords.words('english')
    ps = nltk.stem.PorterStemmer()

    clean_transcript = transcript.lower()
    clean_transcript = nltk.tokenize.word_tokenize(clean_transcript)
    # Remove digits
    clean_transcript = [i for i in clean_transcript if not re.match(r'\d+', i)]
    # Remove Stopwords and single characters
    clean_transcript = [i for i in clean_transcript if i not in stopW and len(i) > 1]
    # Stemming
    clean_transcript = [ps.stem(word) for word in clean_transcript]
    clean_transcript = " ".join(clean_transcript)

    bigrams = list(nltk.bigrams(clean_transcript.split()))
    for bigram in bigrams:
        bigr = bigram[0] + '.' + bigram[1]
        if bigr in political_bigrams:
            return 1

    return 0


def mark_political_news(year, month, db_path):
    curr_path = os.path.dirname(os.path.join(os.path.abspath(__file__)))
    POLITICAL_BIGRAMS = helpers.load_pickle(os.path.join(curr_path, 'data/political_bigrams.pkl'))
    conn = db.NewsDb(db_path)
    print('getting articles from the database...')
    articles = list(conn.select_articles_by_year_and_month(year, month))
    print('checking if article is political')
    articles_polarity = parmap.map(is_political, articles, POLITICAL_BIGRAMS, pm_pbar=True)

    print('updating it to database')
    conn.update_parliament_for_articles(articles, articles_polarity)
    conn.close()


def mark_political_news_for_year(year, db_path):
    for month in range(1, 13):
        mark_political_news(year, month, db_path)


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

    overall_positions = {keyword: [] for keyword in keywords}
    for index, row in df.iterrows():
        bigrams = get_bigrams(row['Transcript'])
        n_bigrams = len(bigrams)
        positions = {keyword: [] for keyword in keywords}

        for i, x in enumerate(bigrams):
            if x not in keywords:
                continue
            positions[x].append(i)

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
    for topic in helpers.topics_id:
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
    topics_df['topic'] = topics_df['topic'].apply(lambda x: helpers.topics_id_to_name_map['02'] if x is np.nan else
    helpers.topics_id_to_name_map[x])
    topics_df.to_csv(os.path.join(path, 'bigram_topics.csv'))
