from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation
from gensim.parsing.preprocessing import remove_stopwords, stem_text, strip_non_alphanum, strip_multiple_whitespaces
from gensim.parsing.preprocessing import strip_short, strip_numeric
from gensim.utils import lemmatize
from dateutil import parser as date_parser
import pickle
import pandas as pd
from copy import copy
import os
import json
import db

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

source_names = ['Sun', 'Mirror', 'Belfast Telegraph', 'Record', 'Independent', 'Observer', 'Guardian', 'People',
                'Telegraph', 'Mail', 'Express', 'Post', 'Herald', 'Star', 'Wales', 'Scotland', 'Standard', 'Scotsman']

source_ids = ['400553', '377101', '418973', '244365', '8200', '412338', '138794', '232241', '334988', '331369',
              '138620', '419001', '8010', '142728', '408506', '143296', '363952', '145251', '232240', '145253',
              '389195', '145254', '344305', '8109', '397135', '163795', '412334', '408508', '411938']

id_to_name_map = {
    '400553': 'Belfast Telegraph',
    '377101': 'Scotsman',
    '418973': 'Record',
    '244365': 'Wales',
    '8200': 'Independent',
    '412338': 'Wales',
    '138794': 'Mail',
    '232241': 'Express',
    '334988': 'Telegraph',
    '331369': 'Sun',
    '138620': 'Guardian',
    '419001': 'Mirror',
    '8010': 'Guardian',
    '142728': 'Herald',
    '408506': 'Express',
    '143296': 'Observer',
    '363952': 'Star',
    '145251': 'People',
    '232240': 'Express',
    '145253': 'Record',
    '389195': 'Telegraph',
    '145254': 'Mirror',
    '344305': 'Scotland',
    '8109': 'Telegraph',
    '397135': 'Mail',
    '163795': 'Belfast Telegraph',
    '412334': 'Post',
    '408508': 'Star',
    '411938': 'Standard'
}

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


def lemma(text):
    out = [wd.decode('utf-8').split('/')[0] for wd in lemmatize(text)]

    return ' '.join(out)


def save(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


def save_json(obj, name):
    with open(name, 'w') as f:
        json.dump(obj, f)


def load_json(name):
    with open(name) as f:
        data = json.load(f)
    return data


def parse_date(date):
    date = date.split('\n')[0]

    if '/' in date:
        date = date.split('/')[1]
    elif '-' in date:
        date = date.split('-')[1]
    elif date.count(',') == 2:
        date = date.split(',')[1]

    if 'juliet' in date:
        date = date.replace('juliet', 'july')

    return date_parser.parse(date)


def preprocess_text(text):
    CLUSTER_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, remove_stopwords, stem_text,
                       strip_non_alphanum, strip_multiple_whitespaces]
    return preprocess_string(text, CLUSTER_FILTERS)


def preprocess_text_for_lda(text):
    LDA_FILTERS = [lambda x: x.lower(), strip_multiple_whitespaces, strip_tags, strip_punctuation, remove_stopwords,
                   strip_short, strip_non_alphanum, strip_numeric]
    return preprocess_string(text, LDA_FILTERS)


def raw_topics_data_to_db(root, db_path='../articles.db'):
    topics = os.listdir(root)
    topics = [topic for topic in topics if (topic.endswith('csv') and not topic.startswith('00')
                                            and not topic.startswith('01') and not topic.startswith('22'))]

    conn = db.NewsDb(db_path)
    for topic in topics:
        print('insert into db rows from {}'.format(topic))
        path = os.path.join(root, topic)
        topic_df = pd.read_csv(path)

        _id = topic.split('_')[0]
        topics = topic_df.apply(lambda row: (_id, topics_id_to_name_map[_id], row['MPs'], row['bigrams'],
                                             row['freq']), axis=1).tolist()
        conn.insert_topics(topics)

    conn.close()


def raw_data_to_db(root, db_path='../articles.db'):
    """Inserts raw news articles data from csv files to a sqlite3 database after preprocessing.
    Parameters
    ----------
    @root: (String) path to root directory where news articles are stored in csv

    Returns
    -------
    None
    """
    newspapers_id = os.listdir(root)
    conn = db.NewsDb(db_path)

    for newspaper in newspapers_id:
        newspaper_path = os.path.join(root, newspaper)
        years = os.listdir(newspaper_path)
        years = [year for year in years if not year.endswith('.pkl')]

        for year in years:
            year_path = os.path.join(newspaper_path, year)

            for month in months:
                month_path = os.path.join(year_path, ' {}-{} {}.csv'.format(month, year, newspaper))
                if os.path.exists(month_path):
                    print(month_path)
                    df = pd.read_csv(month_path)
                    df = df.drop(df.columns[0], axis=1)  # drop the index column
                    df = df.dropna()  # drop rows with column value as nan

                    # make column names consistent across csv
                    df.columns = ['Source', 'Date', 'Program Name', 'Transcript']
                    # change source name to be consistent with it's id
                    df['Source'] = id_to_name_map[newspaper]
                    # drop rows where Transcript is empty string
                    df = df[(df['Program Name'] != "") | (df['Transcript'] != "")]

                    if df.empty:
                        continue

                    df['Date'] = df.apply(lambda row: parse_date(row['Date']), axis=1)  # date string to date object
                    df['Source'] = df.apply(lambda row: row['Source'].strip('.\n '), axis=1)
                    df['Transcript'] = df.apply(lambda row: row['Transcript'].strip(), axis=1)

                    articles = df.apply(lambda row: (
                        newspaper, row['Source'], row['Date'].day, row['Date'].month, row['Date'].year,
                        row['Program Name'], row['Transcript']), axis=1).tolist()

                    conn.insert_articles(articles)
                else:
                    print("{} doesn't exists. Missing data!!".format(month_path))
    conn.close()


def combine_two_dictionary(dct1, dct2):
    """
    Parameters
    ----------
    @dct1: dictionary object
    @dct2: dictionary object

    Returns
    -------
    a dictionary with key from dct1 and dct2 and values from common keys are summed
    """

    combined_dct = copy(dct1)

    for key, val in dct2.items():
        if key in combined_dct:
            combined_dct[key] += val
        else:
            combined_dct[key] = val

    return combined_dct


def combine_list_of_dictionary(dict_list):
    """
    Parameters
    ----------
    @dct: a list of dictionaries

    Returns
    -------
    a single dictionary with common key elements summed
    """

    ret = dict_list[0]

    for i in range(1, len(dict_list)):
        ret = combine_two_dictionary(ret, dict_list[i])

    return ret


def flatten(list_of_lists):
    flat_list = [item for sublist in list_of_lists for item in sublist]
    return flat_list


def remove_stemmed_phrases(s):
    STOPWORDS = []
    phrases = pd.read_csv('../data/alpha_betas_party.csv')['bigram'].tolist()

    for phrase in phrases:
        STOPWORDS.extend(phrase.split('_'))

    return " ".join(w for w in s.split() if w not in STOPWORDS)
