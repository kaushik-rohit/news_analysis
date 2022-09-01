threshold=(0.1 0.2 0.4 0.5 0.6 0.7 0.8 0.9)

for i in "${threshold[@]}";do
    python -m clustering.cluster_analysis -d ../data/news.db -y 2015 -dict ../models/vocab_2015.dict -tf ../models/tfidf_2015 -t "$i"
    python -m clustering.cluster_analysis -d ../articles.db -y 2014 -m 1 -dict ../models/vocab_2014_1.dict -tf ../models/tfidf_2014_1 -t "$i"
done

python -m topics.classifier --doc2vec ./topics/models/doc2vec/doc2vec_12_13 -d ../data/news.db --classifier ./topics/models/doc2vec/classifier_12_13 - y 2014