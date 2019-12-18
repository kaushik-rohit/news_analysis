from dateutil import parser as date_parser
import pandas as pd
import os
import pickle
import db

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']


def save(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f)


def load_data(path):
    newspapers = os.listdir(path)
    df_daywise = []

    for newspaper in newspapers:
        year_2014 = os.path.join(os.path.join(path, newspaper), '2014')
        df_path = os.path.join(year_2014, ' jan-2014 {}.csv'.format(newspaper))
        df = pd.read_csv(df_path)
        df = df.drop(df.columns[0], axis=1)
        df = df.dropna()

        # rename columns so it is consistent across data
        df.columns = ['Source', 'Date', 'Program Name', 'Transcript']
        df = df[(df['Program Name'] != "") & (df['Transcript'] != '')]

        # parse date
        df['Date'] = df.apply(lambda row: row['Date'].split('\n')[0], axis=1)
        df['Date'] = df.apply(lambda row: date_parser.parse(row['Date']), axis=1)

        df_daywise.append(df)

    df_out = pd.concat(df_daywise, axis=0, ignore_index=True).reset_index()
    df_out.to_pickle('./Jan_2014.pkl')
    return df_out


def parse_date(date):
    date = date.split('\n')[0]

    if '/' in date:
        date = date.split('/')[1]
    elif '-' in date:
        date = date.split('-')[1]

    return date_parser.parse(date)


def raw_data_to_db(root):
    newspapers_id = os.listdir(root)
    conn = db.ArticlesDb('../articles.db')

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
                    df = df.drop(df.columns[0], axis=1)
                    df = df.dropna()
                    df.columns = ['Source', 'Date', 'Program Name', 'Transcript']
                    df = df[(df['Program Name'] != "") | (df['Transcript'] != "")]

                    if df.empty:
                        continue

                    df['Date'] = df.apply(lambda row: parse_date(row['Date']), axis=1)
                    df['Source'] = df.apply(lambda row: row['Source'].strip('.\n '), axis=1)
                    df['Transcript'] = df.apply(lambda row: row['Transcript'].strip(), axis=1)

                    articles = df.apply(lambda row: (
                        row['Source'], row['Date'].day, row['Date'].month, row['Date'].year, row['Program Name'],
                        row['Transcript']), axis=1).tolist()

                    conn.insert_articles(articles)
