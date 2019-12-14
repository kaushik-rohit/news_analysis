from datetime import date
from gensim.parsing.preprocessing import preprocess_string


class Article():

    def __init__(self, source, date, program_name, transcript):
        self.source = source
        self.date = date
        self.program_name = program_name
        self.transcript = transcript

    def get_source(self):
        return self.source

    def get_date(self):
        return self.date

    def get_program_name(self):
        return self.program_name

    def get_transcript(self):
        return self.transcript

    def equals(self, article):
        if (self.source == article.get_source() and
        self.date == article.get_date() and
        self.program_name == article.get_program_name):
            return True

        return False


class CorpusIter(object):

    def __init__(self, docs):
        self.docs = iter(docs)

    def __iter__(self):
        for doc in self.docs:
            yield preprocess_string(doc.get_transcript())


class BoWIter(object):

    def __init__(self, dictionary, docs):
        self.dict = dictionary
        self.docs = iter(docs)

    def __iter__(self):
        for doc in self.docs:
            bow = self.dict.doc2bow(preprocess_string(doc.get_transcript()))

            yield bow
