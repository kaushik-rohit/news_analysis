from shared import helpers
import pandas as pd
import numpy as np
import argparse
import calendar
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


def score(start_pos, end_pos):
    return np.sum(range(start_pos, end_pos))/(end_pos - start_pos + 1)


def get_topic_ranks():
    topics = list(helpers.topics_name_to_index_map.keys())
    header = ['day', 'month']
    header += topics
    rows = []
    lrows = []

    for month in range(1, 13):
        for day in range(1, calendar.monthrange(year, month)[1]):  # calendar.monthrange(year, month)[1]
            date_string = '{}-{}-{}'.format(day, calendar.month_abbr[month].lower(), year)
            print(date_string)
            df_day = df.loc[df.date == date_string].reset_index(drop=True)
            partition_ids = df_day.partition_id.unique()
            topic_ranks_for_day = {topic: [] for topic in topics}
            lengths_by_topic = {topic: [] for topic in topics}

            for id_ in partition_ids:
                df_partition = df_day.loc[df_day.partition_id == id_].reset_index(drop=True)
                length = df_partition['length'].sum()
                last_topic = None
                start_pos = 1
                end_pos = 1
                total_length_now = 0

                for index, row in df_partition.iterrows():

                    if total_length_now >= 500:
                        break

                    if last_topic is None:
                        start_pos = 1
                        end_pos = start_pos + row['length'] - 1
                        last_topic = row['topic']
                    elif last_topic == row['topic']:
                        end_pos = end_pos + row['length']
                        if index == len(df_partition) - 1:
                            topic_ranks_for_day[last_topic].append(score(start_pos, end_pos))
                            lengths_by_topic[last_topic].append(end_pos - start_pos + 1)
                    else:
                        topic_ranks_for_day[last_topic].append(score(start_pos, end_pos))
                        lengths_by_topic[last_topic].append(end_pos - start_pos + 1)
                        start_pos = end_pos + 1
                        end_pos = start_pos + row['length'] - 1
                        last_topic = row['topic']
                    total_length_now += row['length']
            row = [day, month]
            lrow = [day, month]
            lrow += [np.mean(lengths_by_topic[topic]) for topic in topics]
            row += [np.mean(topic_ranks_for_day[topic]) for topic in topics]
            rows.append(row)
            lrows.append(lrow)
    res = pd.DataFrame(rows, columns=header)
    res = res.fillna(0)
    res.to_csv('partition_topic_ranks_by_day_month_words_weight_{}_{}.csv'.format(bbc_id, year))

    for topic in topics:
        res[topic] = 500 - res[topic]
    res2 = res.groupby(['month']).agg('mean').drop(['day'], axis=1)
    res2.to_csv('partition_topic_ranks_by_month_words_weight_{}_{}.csv'.format(bbc_id, year))


if __name__ == '__main__':
    pass
