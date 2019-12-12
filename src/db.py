import sqlite3
from sqlite3 import Error
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

create_table_query = ("create table if not exists articles "
                     "(source TEXT, "
                     "day INTEGER, "
                     "month INTEGER, "
                     "year INTEGER, "
                     "program_name TEXT, "
                     "transcript TEXT, "
                     "PRIMARY KEY (source, day, month, year, program_name));")

insert_table_query = ("insert or replace into articles(source, day, month, year, program_name, "
                      "transcript) values(?, ?, ?, ?, ?, ?);")

select_articles_by_year_query = "select * from articles where year=?"

select_articles_by_month_query = "select * from articles where month=?"

select_articles_by_date_query = "select * from articles where day=? and month=? and year=?"


class articles_database():

    def __init__(self, path):
        self.conn = None
        self.path = path

        try:
            self.conn = sqlite3.connect(path)
        except Error as e:
            print(e)

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

    def select_articles_by_month(self, month):
        cur = self.conn.cursor()
        rows = cur.execute(select_articles_by_month_query, (month,))

        return ResultIterator(rows)

    def select_articles_by_date(self, date):
        cur = self.conn.cursor()
        cur.execute(select_articles_by_date_query, (date.day, date.month, date.year))
        rows = cur.fetchall()

        return ResultIterator(rows)

    def _create_table(self):
        try:
            cur = self.conn.cursor()
            cur.execute(create_table_query)
            self.conn.commit()
        except Error as e:
            print(e)


class ResultIterator():

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        for row in self.rows:
            yield simple_preprocess(remove_stopwords(row[5]))

