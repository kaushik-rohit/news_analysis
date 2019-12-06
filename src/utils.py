from dateutil import parser as date_parser
import pandas as pd
import os
import pickle

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
        
        #rename columns so it is consistent across data
        df.columns = ['Source', 'Date', 'Program Name', 'Transcript']
        df = df[(df['Program Name'] != "") & (df['Transcript'] != '')]
        
        #parse date
        df['Date'] = df.apply(lambda row: row['Date'].split('\n')[0], axis=1)
        df['Date'] = df.apply(lambda row: date_parser.parse(row['Date']), axis=1)
        
        df_daywise.append(df)

    return pd.concat(df_daywise, axis=0, ignore_index=True).reset_index()
