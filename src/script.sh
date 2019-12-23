threshold=(0.1 0.2 0.4 0.5 0.6 0.7 0.8 0.9)

for i in "${threshold[@]}";do
    python analysis_gensim.py -d ../articles.db -y 2014 -dict ../models/vocab_2014.dict -tf ../models/tfidf_2014 -t "$i"
    python analysis_gensim.py -d ../articles.db -y 2014 -m 1 -dict ../models/vocab_2014_1.dict -tf ../models/tfidf_2014_1 -t "$i"
done
