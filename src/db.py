import sqlite3
from sqlite3 import Error
from models import Article, Topic
from datetime import date
import helpers

# create queries
create_articles_table_query = ("create table if not exists articles "
                               "(source_id TEXT, "
                               "source TEXT, "
                               "day INTEGER, "
                               "month INTEGER, "
                               "year INTEGER, "
                               "program_name TEXT, "
                               "transcript TEXT, "
                               "PRIMARY KEY (source_id, day, month, year, program_name));")

create_topics_table_query = ("create table if not exists topics "
                             "(id TEXT, "
                             "topic TEXT, "
                             "MP TEXT, "
                             "bigram TEXT, "
                             "frequency INTEGER, "
                             "PRIMARY KEY (id, MP, bigram));")

# insert queries
insert_article_query = ("insert or ignore into articles(source_id, source, day, month, year, program_name, "
                        "transcript) values(?, ?, ?, ?, ?, ?, ?);")

insert_topic_query = "insert or ignore into topics(id, topic, MP, bigram, frequency) values(?, ?, ?, ?, ?);"

# select articles queries
select_articles_by_year_query = "select * from articles where year=?"

select_articles_by_month_query = "select * from articles where year=? and month=?"

select_articles_by_source_and_month_query = "select * from articles where year=? and month=? and source=?"

select_articles_by_date_query = "select * from articles where year=? and month=? and day=?"

select_articles_by_source_and_date_query = "select * from articles where year=? and month=? and day=? and source=?"

select_articles_by_diff_source_and_date_query = ("select * from articles "
                                                 "where year=? and month=? and day=? and source!=?")

get_count_by_year_query = "select count(*) from articles where year=?"

get_count_by_month_and_year_query = "select count(*) from articles where year=? and month=?"

get_count_by_date_query = "select count(*) from articles where year=? and month=? and day=?"

get_all_distinct_source_query = "select distinct(source) from articles;"

get_distinct_source_for_month_query = "select distinct(source) from articles where year=? and month=?"

get_count_by_date_and_source_query = ("select source, count(*) as total_articles from articles "
                                      "where year=? and month=? and day=? group by source")

get_count_by_date_and_source_id_query = ("select source_id, count(*) as total_articles from articles "
                                         "where year=? and month=? and day=? group by source_id")

update_article_topic = ("update articles set topic=? where source_id=? and day=? and month=? and year=? "
                        "and program_name=?")

# select topics queries
select_top_bigrams_for_topic = ("select bigram, sum(frequency) as freq from topics where topic=? "
                                "group by id, topic, bigram order by freq desc limit ?")

select_top_bigrams = "select bigram, sum(frequency) as freq from topics group by bigram order by freq desc limit ?"

select_bigram_freq_for_topic = "select bigram, sum(frequency) as freq from topics where id=? group by bigram"


class NewsDb:
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
            cur.executemany(insert_article_query, articles)
            self.conn.commit()
        except Error as e:
            print(e)

    def insert_topics(self, topics):
        try:
            cur = self.conn.cursor()
            cur.executemany(insert_topic_query, topics)
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

    def select_articles_by_source_and_month(self, year, month, source):
        cur = self.conn.cursor()
        rows = cur.execute(select_articles_by_source_and_month_query, (year, month, source))

        return iter(ResultIterator(rows))

    def select_articles_by_source_and_date(self, source, dt):
        cur = self.conn.cursor()
        rows = cur.execute(select_articles_by_source_and_date_query, (dt.year, dt.month, dt.day, source))

        return iter(ResultIterator(rows))

    def select_articles_by_diff_source_and_date(self, source, dt):
        cur = self.conn.cursor()
        rows = cur.execute(select_articles_by_diff_source_and_date_query, (dt.year, dt.month, dt.day, source))

        return iter(ResultIterator(rows))

    def select_articles_by_date(self, dt):
        cur = self.conn.cursor()
        rows = cur.execute(select_articles_by_date_query, (dt.year, dt.month, dt.day))

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

    def get_count_of_articles_for_date(self, dt):
        cur = self.conn.cursor()
        cur.execute(get_count_by_date_query, (dt.year, dt.month, dt.day))

        return cur.fetchone()[0]

    def get_count_of_articles_for_date_by_source(self, dt):
        cur = self.conn.cursor()
        cur.execute(get_count_by_date_and_source_query, (dt.year, dt.month, dt.day))

        return cur.fetchall()

    def get_count_of_articles_for_date_by_source_id(self, dt):
        cur = self.conn.cursor()
        cur.execute(get_count_by_date_and_source_id_query, (dt.year, dt.month, dt.day))

        return cur.fetchall()

    def get_all_new_source(self):
        cur = self.conn.cursor()
        cur.execute(get_all_distinct_source_query)

        return cur.fetchall()

    def get_news_source_for_month(self, year, month):
        cur = self.conn.cursor()
        cur.execute(get_distinct_source_for_month_query, (year, month))

        return cur.fetchall()

    def update_topic_for_articles(self, articles, topics):
        assert (len(articles) == len(topics))
        n = len(articles)
        params = []

        for i in range(n):
            article = articles[i]
            topic = topics[i]
            params.append((topic, article.source_id, article.date.day, article.date.month, article.date.year,
                           article.program_name))
        try:
            cur = self.conn.cursor()
            cur.executemany(update_article_topic, params)
            self.conn.commit()
        except Error as e:
            print(e)

    def get_bigram_freq_for_topic(self, topic_id):
        bigrams_freq = {}
        cur = self.conn.cursor()
        cur.execute(select_bigram_freq_for_topic, (topic_id,))
        rows = cur.fetchall()

        for row in rows:
            bigrams_freq[row[0]] = row[1]
        return bigrams_freq

    def get_corpus_for_topic(self, topic_id, bigrams):
        corpus = []
        freqs = self.get_bigram_freq_for_topic(topic_id)
        for bigram in bigrams:
            if bigram in freqs:
                corpus += [bigram] * freqs[bigram]
        return corpus

    def select_top_bigrams(self, top_n):
        bigrams_freq = []
        cur = self.conn.cursor()
        cur.execute(select_top_bigrams, (top_n,))
        rows = cur.fetchall()

        for row in rows:
            bigrams_freq += [row[0]]

        return bigrams_freq

    def select_n_top_bigrams_from_each_topic(self, top_n):
        bigrams = []
        for topic in helpers.topics_id:
            cur = self.conn.cursor()
            cur.execute(select_top_bigrams_for_topic, (topic, top_n, ))
            rows = cur.fetchall()

            for row in rows:
                bigrams += [row[0]]
        return bigrams

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
            cur.execute(create_articles_table_query)
            cur.execute(create_topics_table_query)
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
            yield Article(row[0], row[1], date(row[4], row[3], row[2]), row[5], row[6], row[7])
