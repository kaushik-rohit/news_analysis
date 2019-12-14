import sqlite3
from sqlite3 import Error
from models import Article
from datetime import date
from copy import copy

create_table_query = ("create table if not exists articles "
                     "(source TEXT, "
                     "day INTEGER, "
                     "month INTEGER, "
                     "year INTEGER, "
                     "program_name TEXT, "
                     "transcript TEXT, "
                     "PRIMARY KEY (source, day, month, year, program_name));")

insert_table_query = ("insert on conflict ignore into articles(source, day, month, year, program_name, "
                      "transcript) values(?, ?, ?, ?, ?, ?);")

select_articles_by_year_query = "select * from articles where year=?"

select_articles_by_month_query = "select * from articles where year=? and month=?"

select_articles_by_date_query = "select * from articles where day=? and month=? and year=?"

get_count_by_year_query = "select count(*) from articles where year=?"

get_count_by_month_and_year_query = "select count(*) from articles where year=? and month=?"

get_count_by_date_query = "select count(*) from articles where day=? and month=? and year=?"

get_all_distinct_source_query = "select distinct(source) from articles;"


class articles_database():

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

        return ResultIterator(rows)

    def select_articles_by_year_and_month(self, year, month):
        cur = self.conn.cursor()
        rows = cur.execute(select_articles_by_month_query, (year, month,))

        return ResultIterator(rows)

    def select_articles_by_date(self, date):
        cur = self.conn.cursor()
        rows = cur.execute(select_articles_by_date_query, (date.day, date.month, date.year))

        return ResultIterator(rows)

    def get_count_of_articles_for_year_and_month(self, year, month):
        try:
            cur = self.conn.cursor()
            cur.execute(get_count_by_month_and_year_query, (year, month),)
        except Error as e:
            print(e)

        return cur.fetchone()[0]

    def get_count_of_articles_for_year(self, year):
        cur = self.conn.cursor()
        cur.execute(get_count_by_year_query, (year,))

        return cur.fetchone()[0]

    def get_count_of_articles_for_date(self, date):
        cur = self.conn.cursor()
        cur.execute(get_count_by_date_query, (date.day, date.month, date.year))

        return cur.fetchone()[0]

    def get_all_new_source(self):
        cur = self.conn.cursor()
        cur.execute()

        return cur.fetchall()

    def _get_connection(self, path):
        if self.conn != None:
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

class ResultIterator():

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        for row in self.rows:
            yield Article(row[0], date(row[3], row[2], row[1]), row[4], row[5])

