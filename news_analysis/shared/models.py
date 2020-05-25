class Article:

    def __init__(self, source_id, source, date, program_name, transcript, topic):
        self.source_id = source_id
        self.source = source
        self.date = date
        self.program_name = program_name
        self.transcript = transcript
        self.bigrams = None
        self.topic = topic

    def __str__(self):
        return 'article id: {}, name: {}, date: {}, program_name: {}'.format(self.source_id, self.source, self.date,
                                                                             self.program_name)

    def __eq__(self, other):
        if isinstance(other, Article):
            return ((self.source == other.source and
                     self.source_id == other.source_id and
                     self.date == other.date and
                     self.program_name == other.program_name))
        else:
            return False

    def equals(self, article):
        if (self.source == article.source and
                self.date == article.date and
                self.program_name == article.program_name):
            return True

        return False


class Topic:

    def __init__(self, _id, topic, MP, bigram, frequency):
        self.id = _id
        self.topic = topic
        self.MP = MP
        self.bigram = bigram
        self.frequency = frequency

    def __str__(self):
        return 'topic id: {}, used by: {}, bigram: {}, frequency: {}'.format(self.id, self.MP, self.bigram,
                                                                             self.frequency)


class CorpusIter(object):

    def __init__(self, docs, preprocess_fn=None, iterate_on='transcript'):
        self.docs = docs
        self.preprocess_fn = preprocess_fn
        self.iterate_on = iterate_on

    def __iter__(self):
        for doc in self.docs:
            if self.preprocess_fn is None:
                if self.iterate_on == 'transcript':
                    yield doc.transcript
                elif self.iterate_on == 'program_name':
                    yield doc.program_name
            else:
                if self.iterate_on == 'transcript':
                    yield self.preprocess_fn(doc.transcript)
                elif self.iterate_on == 'program_name':
                    yield self.preprocess_fn(doc.program_name)


class BoWIter(object):

    def __init__(self, dictionary, docs, preprocess_fn=None, bigram=None, iterate_on='transcript'):
        self.dict = dictionary
        self.docs = docs
        self.preprocess_fn = preprocess_fn
        self.bigram = bigram
        self.iterate_on = iterate_on

    def __iter__(self):
        for doc in self.docs:
            if self.preprocess_fn is None:
                if self.iterate_on == 'transcript':
                    bow = self.dict.doc2bow(doc.transcript)
                elif self.iterate_on == 'program_name':
                    bow = self.dict.doc2bow(doc.program_name)
            elif self.bigram is None:
                if self.iterate_on == 'transcript':
                    bow = self.dict.doc2bow(self.preprocess_fn(doc.transcript))
                elif self.iterate_on == 'program_name':
                    bow = self.dict.doc2bow(self.preprocess_fn(doc.program_name))
            else:
                if self.iterate_on == 'transcript':
                    bow = self.dict.doc2bow(self.bigram[self.preprocess_fn(doc.transcript)])
                elif self.iterate_on == 'program_name':
                    bow = self.dict.doc2bow(self.bigram[self.preprocess_fn(doc.program_name)])

            yield bow
