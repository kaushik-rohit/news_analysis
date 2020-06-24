import unittest
from clustering import cluster_analysis


class FakeArticle:

    def __init__(self, source):
        self.source = source


class TestClusterAnalysis(unittest.TestCase):

    def test_get_pos_of_same_source_news(self):
        corpus = [FakeArticle('Guardian'), FakeArticle('Independent'), FakeArticle('Guardian'), FakeArticle('Mirror')]
        res = cluster_analysis.get_pos_of_same_source_news(corpus)
        print(res)

    def test_get_pos_of_same_source_news2(self):
        corpus1 = [FakeArticle('Guardian'), FakeArticle('Independent'), FakeArticle('Guardian'), FakeArticle('Mirror')]
        corpus2 = [FakeArticle('Independent'), FakeArticle('Mirror'), FakeArticle('Mirror'), FakeArticle('Guardian')]
        res = cluster_analysis.get_pos_of_same_source_news(corpus1, corpus2)
        print(res)


if __name__ == '__main__':
    unittest.main()
