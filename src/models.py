from gensim.parsing.preprocessing import preprocess_string, strip_tags, strip_punctuation
from gensim.parsing.preprocessing import remove_stopwords, stem_text, strip_non_alphanum, strip_multiple_whitespaces
import helpers

CUSTOM_FILTERS = [lambda x: x.lower(), strip_tags, strip_punctuation, remove_stopwords, stem_text,
                  strip_non_alphanum, strip_multiple_whitespaces]

# helpers.remove_stemmed_phrases,


class Article:

    def __init__(self, source_id, source, date, program_name, transcript):
        self.source_id = source_id
        self.source = source
        self.date = date
        self.program_name = program_name
        self.transcript = transcript

    def equals(self, article):
        if (self.source == article.source and
                self.date == article.date and
                self.program_name == article.program_name):
            return True

        return False


class CorpusIter(object):

    def __init__(self, docs):
        self.docs = docs

    def __iter__(self):
        for doc in self.docs:
            yield preprocess_string(doc.transcript, CUSTOM_FILTERS)


class BoWIter(object):

    def __init__(self, dictionary, docs):
        self.dict = dictionary
        self.docs = docs

    def __iter__(self):
        for doc in self.docs:
            bow = self.dict.doc2bow(preprocess_string(doc.transcript, CUSTOM_FILTERS))

            yield bow
