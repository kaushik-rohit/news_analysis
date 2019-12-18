import sqlite3
from sqlite3 import Error
from models import Article
from datetime import date

# create queries
create_table_query = ("create table if not exists articles "
                      "(source TEXT, "
                      "day INTEGER, "
                      "month INTEGER, "
                      "year INTEGER, "
                      "program_name TEXT, "
                      "transcript TEXT, "
                      "PRIMARY KEY (source, day, month, year, program_name));")

# insert queries
insert_table_query = ("insert or ignore into articles(source, day, month, year, program_name, "
                      "transcript) values(?, ?, ?, ?, ?, ?);")


# select queries
select_articles_by_year_query = "select * from articles where year=?"

select_articles_by_month_query = "select * from articles where year=? and month=?"

select_articles_by_date_query = "select * from articles where year=? and month=? and day=?"

select_articles_by_source_and_date_query = "select * from articles where year=? and month=? and day=? and source=?"

select_articles_by_diff_source_and_date_query = ("select * from articles "
                                                 "where year=? and month=? and day=? and source!=?")

get_count_by_year_query = "select count(*) from articles where year=?"

get_count_by_month_and_year_query = "select count(*) from articles where year=? and month=?"

get_count_by_date_query = "select count(*) from articles where year=? and month=? and day=?"

get_all_distinct_source_query = "select distinct(source) from articles;"

get_count_by_date_and_source_query = ("select source, count(*) as total_articles from articles "
                                      "where year=? and month=? and day=? group by source")


class ArticlesDb:
    """
    The Database class that handles the connection to sqlite3 db where news articles are stored.
    It provides some helper functions that can be used to gather stats from the data or perform
    read-write query.
    """
    def __init__(self, path):
        self.conn = None
        self.path = path
        self._get_connection(path)
        self._create_table()

    def insert_articles(self, articles):
        try:
            cur = self.conn.cursor()
            cur.executemany(insert_table_query, articles)
            self.conn.commit()
        except Error as e:
            print(e)

    def select_articles_by_year(self, year):
        cur = self.conn.cursor()
        rows = cur.execute(select_articles_by_year_query, (year,))

        return iter(ResultIterator(rows))

    def select_articles_by_year_and_month(self, year, month):
        cur = self.conn.cursor()
        rows = cur.execute(select_articles_by_month_query, (year, month,))

        return iter(ResultIterator(rows))

    def select_articles_by_source_and_date(self, source, date):
        cur = self.conn.cursor()
        rows = cur.execute(select_articles_by_source_and_date_query, (date.year, date.month, date.day, source))

        return iter(ResultIterator(rows))

    def select_articles_by_diff_source_and_date(self, source, date):
        cur = self.conn.cursor()
        rows = cur.execute(select_articles_by_diff_source_and_date_query, (date.year, date.month, date.day, source))

        return iter(ResultIterator(rows))

    def select_articles_by_date(self, date):
        cur = self.conn.cursor()
        rows = cur.execute(select_articles_by_date_query, (date.year, date.month, date.day))

        return iter(ResultIterator(rows))

    def get_count_of_articles_for_year_and_month(self, year, month):
        try:
            cur = self.conn.cursor()
            cur.execute(get_count_by_month_and_year_query, (year, month), )
        except Error as e:
            print(e)

        return cur.fetchone()[0]

    def get_count_of_articles_for_year(self, year):
        cur = self.conn.cursor()
        cur.execute(get_count_by_year_query, (year,))

        return cur.fetchone()[0]

    def get_count_of_articles_for_date(self, date):
        cur = self.conn.cursor()
        cur.execute(get_count_by_date_query, (date.year, date.month, date.day))

        return cur.fetchone()[0]

    def get_count_of_articles_for_date_by_source(self, date):
        cur = self.conn.cursor()
        cur.execute(get_count_by_date_and_source_query, (date.year, date.month, date.day))

        return cur.fetchall()

    def get_all_new_source(self):
        cur = self.conn.cursor()
        cur.execute(get_all_distinct_source_query)

        return cur.fetchall()

    def _get_connection(self, path):
        if self.conn is not None:
            return

        try:
            self.conn = sqlite3.connect(path)
        except Error as e:
            print(e)

    def _create_table(self):
        try:
            cur = self.conn.cursor()
            cur.execute(create_table_query)
            self.conn.commit()
        except Error as e:
            print(e)

    def close(self):
        self.conn.close()
        self.conn = None


class ResultIterator:

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        for row in self.rows:
            yield Article(row[0], date(row[3], row[2], row[1]), row[4], row[5])
