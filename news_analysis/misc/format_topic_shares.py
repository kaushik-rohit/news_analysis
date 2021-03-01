import pandas as pd


def get_month(source_name):
    parts = source_name.split(' ')

    if len(parts) == 2:
        str_date = parts[0]
    else:
        str_date = parts[1]
    month, _ = str_date.split('-')

    return month


def get_year(source_name):
    parts = source_name.split(' ')

    if len(parts) == 2:
        str_date = parts[0]
    else:
        str_date = parts[1]
    _, year = str_date.split('-')

    return year


def get_topic(source_name):
    parts = source_name.split(' ')

    if len(parts) == 2:
        topic_str = parts[1]
    else:
        topic_str = parts[2]

    return topic_str.split('.csv')[0]

def change_month_name(name):
    month_map = {'jan':1, 'feb':2, 'mar':3,'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}

    return month_map[name]

def do_it():
    df = pd.read_csv('topic_shares.csv')
    df = df.drop(['Unnamed: 0'], axis=1)
    df['month'] = df['source'].apply(lambda x: get_month(x))
    df['year'] = df['source'].apply(lambda x: get_year(x))
    df['topic'] = df['source'].apply(lambda x: get_topic(x)) 
    df = df.drop(['source'], axis=1)
    topics = df.topic.unique()
    years = df.year.unique()
    months = df.month.unique()
    topics.sort()

    shares = {year:{month:{topic:0 for topic in topics} for month in months} for year in years}
    for index, row in df.iterrows(): 
        year = row['year'] 
        month = row['month'] 
        share = row['share'] 
        topic = row['topic'] 
        shares[year][month][topic] = share

    rows = []

    for year in years:
        for month in months:
            row = [year, month]
            for topic in topics:
                row.append(shares[year][month][topic])
            rows.append(row)

    columns = ['year', 'month']

    for topic in topics:
        columns.append(topic)
    
    res_df = pd.DataFrame(rows, columns=columns)

    res_df['month'] = res_df['month'].apply(lambda x: change_month_name(x))
    res_df = res_df.sort_values(by=['year', 'month'])
    res_df.to_csv('topic_shares_matrix_format.csv')

if __name__ == '__main__':
    do_it()
