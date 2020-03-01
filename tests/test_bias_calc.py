import unittest
import pandas as pd
import helpers
import numpy as np


class TestBiasCalculation(unittest.TestCase):

    def test_bigrams_share(self):
        shares_all_articles = pd.read_csv('../tests/data/bigrams_share_all_articles_2015_1.csv')
        shares_in_cluster = pd.read_csv('../tests/data/bigrams_share_in_cluster_2015_1.csv')
        shares_not_in_cluster = pd.read_csv('../tests/data/bigrams_share_not_in_cluster_2015_1.csv')
        bigrams_count_all_articles = helpers.load_json('../tests/data/total_bigrams_all_articles_2015_1.json')
        bigrams_count_in_cluster = helpers.load_json('../tests/data/total_bigrams_in_cluster_2015_1.json')
        bigrams_count_not_in_cluster = helpers.load_json('../tests/data/total_bigrams_not_in_cluster_2015_1.json')

        bigrams = shares_all_articles.columns[2:]
        sources = shares_all_articles['source'].tolist()
        n_source = len(sources)

        for index, row in shares_all_articles.iterrows():
            source = row['source']
            for bigram in bigrams:
                share_in_all_article = row[bigram]
                share_in_cluster = shares_in_cluster.loc[shares_in_cluster['source'] == source].iloc[0][bigram]
                share_not_in_cluster = shares_not_in_cluster.loc[shares_not_in_cluster['source'] ==
                                                                 source].iloc[0][bigram]

                count_in_cluster = bigrams_count_in_cluster[source]
                count_not_in_cluster = bigrams_count_not_in_cluster[source]
                total_count = count_in_cluster + count_not_in_cluster
                weighted_share = (share_in_cluster*count_in_cluster + share_not_in_cluster*count_not_in_cluster)/total_count
                assert(np.isclose(weighted_share, share_in_all_article))


if __name__ == '__main__':
    unittest.main()
