from datetime import date, timedelta
from gensim import corpora, models
import cluster_analysis
import pandas as pd
import helpers
import argparse
import calendar
import db

# create necessary arguments to run the analysis
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--db-path',
                    type=str,
                    required=True,
                    help='the path to database where news articles are stored')

parser.add_argument('-m', '--month',
                    choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    type=int,
                    default=None,
                    help='The month for which analysis is to be performed. If month is not provided '
                         'the analysis is performed on the whole year')

parser.add_argument('-y', '--year',
                    choices=[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
                             2015, 2016, 2017, 2018],
                    type=int,
                    default=2014,
                    help='The year for which analysis is to be performed. By default takes the value 2014')

parser.add_argument('-dict', '--dictionary',
                    type=str,
                    required=True,
                    help='the path to bag of words model')

parser.add_argument('-tf', '--tfidf-model',
                    type=str,
                    required=True,
                    help='the path to trained tfidf gensim model')

parser.add_argument('-t', '--threshold',
                    type=float,
                    default=0.3,
                    help='threshold for cosine similarity to consider news articles in same cluster')

parser.add_argument('-g', '--group-by',
                    type=str,
                    choices=['source_id', 'source_name'],
                    default='source_name',
                    help='whether bias is categorized by source id or source name. Multiple source id can have same'
                         'name because online and print version have different id')


def group_articles(articles):
    group = {source: 0 for source in helpers.source_names}

    for article in articles:
        if article.source in group:
            group[article.source] += 1
        else:
            group[article.source] = 1

    return group


def calculate_tomorrows_cluster_statistics(db_path, dct, tfidf_model, year, month, threshold):
    start_date = date(year, month, 1)
    end_date = date(year, month, calendar.monthrange(year, month)[1])  # calendar.monthrange(year, month)[1]
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    conn = db.NewsDb(db_path)
    stats = []  # date, source, number of articles in tomorrows cluster, total news for this source
    for curr_date in date_range:
        source_count_map = {source: 0 for source in helpers.source_names}
        source_count = conn.get_count_of_articles_for_date_by_source(curr_date)

        for sc in source_count:
            source_count_map[sc[0]] = sc[1]

        _, _, in_tomorrows_cluster = cluster_analysis.get_cluster_for_the_day(curr_date, db_path,
                                                                              dct, tfidf_model,
                                                                              threshold)
        counts_by_source_in_tomorrows_cluster = group_articles(in_tomorrows_cluster)
        for source in helpers.source_names:
            stats += [{'date': curr_date, 'source': source,
                       'articles_in_tomorrows_cluster': counts_by_source_in_tomorrows_cluster[source],
                       'total_news_articles': source_count_map[source]}]
    conn.close()
    df = pd.DataFrame(stats)
    df.to_csv(path_or_buf='../results/stats_tomorrows_articles.csv')


def main():
    args = parser.parse_args()

    dct = corpora.Dictionary.load(args.dictionary)
    tfidf_model = models.TfidfModel.load(args.tfidf_model)
    year = args.year
    month = args.month
    threshold = args.threshold
    db_path = args.db_path

    calculate_tomorrows_cluster_statistics(db_path, dct, tfidf_model, year, month, threshold)


if __name__ == '__main__':
    main()
