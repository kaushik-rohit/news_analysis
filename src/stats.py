from datetime import date, timedelta
from gensim import corpora, models
import multiprocessing as mp
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


def group_articles_by_source_name(articles):
    group = {source: 0 for source in helpers.source_names}

    for article in articles:
        group[article.source] += 1

    return group


def group_articles_by_source_id(articles):
    group = {_id: 0 for _id in helpers.source_ids}

    for article in articles:
        group[article.source_id] += 1

    return group


def get_news_count_by_source(db_path, curr_date):
    conn = db.NewsDb(db_path)

    source_count_id_map = {source_id: 0 for source_id in helpers.source_ids}
    source_count_id = conn.get_count_of_articles_for_date_by_source_id(curr_date)
    source_count_map = {source: 0 for source in helpers.source_names}
    source_count = conn.get_count_of_articles_for_date_by_source(curr_date)

    for sc in source_count_id:
        source_count_id_map[sc[0]] = sc[1]

    for sc in source_count:
        source_count_map[sc[0]] = sc[1]

    conn.close()

    return source_count_id_map, source_count_map


def format_results_into_rows(db_path, date_range, cluster_map):
    rows_source_name = []
    rows_source_id = []
    for curr_date in date_range:
        next_date = curr_date + timedelta(days=1)
        cluster_articles = cluster_map[curr_date]
        source_count_id_map, source_count_map = get_news_count_by_source(db_path, curr_date)
        source_count_id_map_next, source_count_map_next = get_news_count_by_source(db_path, next_date)

        counts_by_source_in_tomorrows_cluster = group_articles_by_source_name(cluster_articles)
        counts_by_source_id_in_tomorrows_cluster = group_articles_by_source_id(cluster_articles)

        for source in helpers.source_names:
            rows_source_name += [
                {
                    'date': curr_date,
                    'source': source,
                    'tomorrows_article_which_report_on_today': counts_by_source_in_tomorrows_cluster[source],
                    'total_news_articles_today': source_count_map[source],
                    'total_news_articles_tomorrow': source_count_map_next[source]
                }
            ]

        for source_id in helpers.source_ids:
            rows_source_id += [
                {
                    'date': curr_date,
                    'source_id': source_id,
                    'tomorrows_article_which_report_on_today': counts_by_source_id_in_tomorrows_cluster[source_id],
                    'total_news_articles_today': source_count_id_map[source_id],
                    'total_news_articles_tomorrow': source_count_id_map_next[source_id]
                }
            ]

    return rows_source_name, rows_source_id


def format_in_tomorrows_cluster_results_into_rows(db_path, date_range, cluster_map):
    rows_source_name = []
    rows_source_id = []

    for curr_date in date_range:
        cluster_articles = cluster_map[curr_date]
        source_count_id_map, source_count_map = get_news_count_by_source(db_path, curr_date)

        counts_by_source_in_tomorrows_cluster = group_articles_by_source_name(cluster_articles)
        counts_by_source_id_in_tomorrows_cluster = group_articles_by_source_id(cluster_articles)

        for source in helpers.source_names:
            rows_source_name += [
                {
                    'date': curr_date,
                    'source': source,
                    'articles_in_tomorrows_cluster': counts_by_source_in_tomorrows_cluster[source],
                    'total_news_articles': source_count_map[source]
                }
            ]

        for source_id in helpers.source_ids:
            rows_source_id += [
                {
                    'date': curr_date,
                    'source_id': source_id,
                    'articles_in_tomorrows_cluster': counts_by_source_id_in_tomorrows_cluster[source_id],
                    'total_news_articles': source_count_id_map[source_id]
                }
            ]

    return rows_source_name, rows_source_id


def calculate_tomorrows_cluster_statistics(db_path, dct, tfidf_model, year, month, threshold):
    start_date = date(year, month, 1)
    end_date = date(year, month, calendar.monthrange(year, month)[1])  # calendar.monthrange(year, month)[1]
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    pool = mp.Pool(mp.cpu_count())  # calculate stats for date in parallel
    stats = pool.starmap(cluster_analysis.get_cluster_for_the_day_with_tomorrows_articles, [(curr_date, db_path, dct,
                                                                                             tfidf_model, threshold)
                                                                                            for curr_date in
                                                                                            date_range])
    assert (len(stats) == len(date_range))
    in_tomorrows_cluster_map = {}
    tomorrows_article_in_tomorrows_cluster_map = {}

    for i in range(len(date_range)):
        in_tomorrows_cluster_map[date_range[i]] = stats[i][2]
        tomorrows_article_in_tomorrows_cluster_map[date_range[i]] = stats[i][3]

    # save results to csv
    rows_source_name, rows_source_id = format_in_tomorrows_cluster_results_into_rows(db_path, date_range,
                                                                                     in_tomorrows_cluster_map)
    df = pd.DataFrame(rows_source_name)
    df.to_csv(path_or_buf='../results/stats_in_tomorrows_cluster_source_name.csv')

    df = pd.DataFrame(rows_source_id)
    df.to_csv(path_or_buf='../results/stats_in_tomorrows_cluster_source_id.csv')

    rows_source_name, rows_source_id = format_results_into_rows(db_path, date_range,
                                                                tomorrows_article_in_tomorrows_cluster_map)
    df = pd.DataFrame(rows_source_name)
    df.to_csv(path_or_buf='../results/stats_tomorrows_articles_source_name.csv')

    df = pd.DataFrame(rows_source_id)
    df.to_csv(path_or_buf='../results/stats_tomorrows_article_source_id.csv')


def get_maximum_similarities_of_articles_by_source(curr_date, path, dct, tfidf_model, threshold):
    delta = timedelta(days=1)
    next_date = curr_date + delta
    conn = db.NewsDb(path)

    rows = []

    print('calculating clusters for {}'.format(curr_date))

    articles_day1 = list(conn.select_articles_by_date(curr_date))
    articles_day2 = list(conn.select_articles_by_date(next_date))

    conn.close()
    unclustered_articles_indices = cluster_analysis.get_articles_not_in_cluster(articles_day1, dct, tfidf_model,
                                                                                threshold=threshold)
    unclustered_articles = [articles_day1[i] for i in unclustered_articles_indices]
    unclustered_articles_indices_in_day2_cluster = cluster_analysis.get_similar_articles(unclustered_articles,
                                                                                         articles_day2, dct,
                                                                                         tfidf_model, threshold)

    unclustered_articles_in_day2_cluster = [unclustered_articles[i] for i, idx in
                                            unclustered_articles_indices_in_day2_cluster]

    similarities = cluster_analysis.get_similarities(unclustered_articles_in_day2_cluster, articles_day2, dct,
                                                     tfidf_model)

    for idx, sim in similarities:
        max_similarity_by_source = {source: 0 for source in helpers.source_names}
        assert (len(sim) == len(articles_day2))
        article = unclustered_articles_in_day2_cluster[idx]
        for i in range(len(sim)):
            source = articles_day2[i].source
            max_similarity_by_source[source] = max(max_similarity_by_source[source], sim[i])

        row = [curr_date, article.source, article.program_name]
        row += [max_similarity_by_source[source] for source in helpers.source_names]
        rows += [row]

    return rows


def calculate_tomorrow_similarity_statistics(db_path, dct, tfidf_model, year, month, threshold):
    start_date = date(year, month, 1)
    end_date = date(year, month, calendar.monthrange(year, month)[1])  # calendar.monthrange(year, month)[1]
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

    pool = mp.Pool(mp.cpu_count())  # calculate stats for date in parallel
    rows = []

    stats = pool.starmap(get_maximum_similarities_of_articles_by_source, [(curr_date, db_path, dct,
                                                                           tfidf_model, threshold)
                                                                          for curr_date in
                                                                          date_range])

    for stat in stats:
        rows += stat

    columns = ['date', 'source', 'program'] + [source for source in helpers.source_names]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path_or_buf='../results/tomorrow_articles_similarity_by_source.csv')


def main():
    args = parser.parse_args()

    dct = corpora.Dictionary.load(args.dictionary)
    tfidf_model = models.TfidfModel.load(args.tfidf_model)
    year = args.year
    month = args.month
    threshold = args.threshold
    db_path = args.db_path

    calculate_tomorrows_cluster_statistics(db_path, dct, tfidf_model, year, month, threshold)
    # calculate_tomorrow_similarity_statistics(db_path, dct, tfidf_model, year, month, threshold)


if __name__ == '__main__':
    main()
