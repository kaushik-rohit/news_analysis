# NewsAnalysis
News analysis project aims to study the biasness of articles and the effects of biasness. i.e How do different news sources report a news, Whether biasness across newspapers remains same across months, and whether left-biased source picks/reports news from other left-biased source or also from right-biased source.

# Experiments
## 1. News articles reporting
  
  ### 1.1 In Cluster
  
  ### 1.2 Not In Cluster
  
  ### 1.3 In Tomorrow's Cluster
  
## 2. Biasness of articles in different clusters

### 2.1 Bias In Cluster

### 2.2 Bias Not In Cluster

### 2.3 Bias In Tomorrow's Cluster

### 2.4 Bias All Articles

### 2.5 Bias by Source-Source reporting

## 2. Topics
We identify topics for each news articles based on correlation of bigrams between article transcript and 
parliament speeches.

### 2.1 LDA

### 2.2 Doc2Vec

### 2.3 XLNET

## 3. Database
We use sqlite3 for storing articles and topics data and perform fast query operations. Some indexes has to be
introduced to speedup the read operation. The schema of database is presented below:

```CREATE TABLE articles (source_id TEXT, source TEXT, day INTEGER, month INTEGER, year INTEGER, program_name TEXT, transcript TEXT, topic text, PRIMARY KEY (source_id, day, month, year, program_name));
CREATE INDEX date_index on articles(year, month, day, source, source_id);
CREATE INDEX source_index on articles(source);
CREATE INDEX source_id_index on articles(source_id);
CREATE TABLE topics (id TEXT, topic TEXT, MP TEXT, bigram TEXT, frequency INTEGER, PRIMARY KEY (id, MP, bigram));
```
