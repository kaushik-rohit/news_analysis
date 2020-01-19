import argparse
from datetime import date, timedelta
from cluster_analysis import get_articles_not_in_cluster
from helpers import *
import calendar
import multiprocessing as mp

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

top_1000_bigrams = []


def bias_averaged_over_month(db_path, dct, tfidf_model, month, year):
    delta = timedelta(days=1)
    month_name = {1: 'jan', 2: 'feb', 3: 'mar', 4: 'apr', 5: 'may', 6: 'jun', 7: 'jul', 8: 'aug', 9: 'sep', 10: 'oct',
                  11: 'nov', 12: 'dec'}
    start_date = date(year, month, 1)
    end_date = date(year, month, calendar.monthrange(year, month)[1])


def bias_averaged_over_year(db_path, dct, tfidf_model, year, threshold=0.3):
    pass


def calculate_bias(db_path, dct, tfidf_model, date, threshold):
    pass


def main():
    args = parser.parse_args()

    dct = args.dictionary
    tfidf_model = args.tfidf_model
    year = args.year
    month = args.month
    threshold = args.threshold
    db_path = args.db_path

    if month is None:
        bias_averaged_over_year(db_path, dct, tfidf_model, year, threshold=threshold)
    else:
        bias_averaged_over_year(db_path, dct, tfidf_model, year, threshold=threshold)


if __name__ == '__main__':
    pass
