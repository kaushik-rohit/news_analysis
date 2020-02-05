import os
import pickle
import pandas as pd
import json
from dateutil import parser as date_parser
import db
from copy import copy

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
id_to_name_map = {'400553': 'Belfast Telegraph Online',
                  '377101': 'The Scotsman',
                  '418973': 'dailyrecord.co.uk',
                  '244365': 'Wales on Sunday',
                  '8200': 'The Independent (United Kingdom)',
                  '412338': 'walesonline.co.uk',
                  '138794': 'DAILYMAIL',
                  '232241': 'Sunday Express',
                  '334988': 'The Sunday Telegraph (London)',
                  '331369': 'Sunday Sun',
                  '138620': 'The Guardian',
                  '419001': 'mirror.co.uk',
                  '8010': 'Guardian Weekly',
                  '142728': 'The Herald (Glasgow)',
                  '408506': 'Express Online',
                  '143296': 'The Observer (London)',
                  '363952': 'Daily Star Sunday',
                  '145251': 'The People',
                  '232240': 'The Express',
                  '145253': 'Daily Record',
                  '389195': 'telegraph.co.uk',
                  '145254': 'The Mirror',
                  '344305': 'Scotland on Sunday',
                  '8109': 'The Daily Telegraph (London)',
                  '397135': 'MailOnline',
                  '163795': 'Belfast Telegraph',
                  '412334': 'dailypost.co.uk',
                  '408508': 'Daily Star Online',
                  '411938': 'standard.co.uk'}


def save(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


def save_json(obj, name):
    with open(name, 'wb') as f:
        json.dump(obj, f)


def parse_date(date):
    date = date.split('\n')[0]

    if '/' in date:
        date = date.split('/')[1]
    elif '-' in date:
        date = date.split('-')[1]

    return date_parser.parse(date)


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
    conn = db.ArticlesDb(db_path)

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
                        row['Source'], row['Date'].day, row['Date'].month, row['Date'].year, row['Program Name'],
                        row['Transcript']), axis=1).tolist()

                    conn.insert_articles(articles)
                else:
                    print("{} doesn't exists. Missing data!!".format(month_path))


def combine_dct(dct1, dct2):
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


def combine_dictionaries(dct):
    """
    Parameters
    ----------
    @dct: a list of dictionaries

    Returns
    -------
    a single dictionary with common key elements summed
    """

    ret = dct[0]

    for i in range(1, len(dct)):
        ret = combine_dct(ret, dct[i])

    return ret
