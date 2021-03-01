import pandas as pd
import os

files = os.listdir('.')
files = [_file for _file in files if _file.endswith('.csv')]
mp_bigram_count = {}

for _file in files: 
        df = pd.read_csv(_file) 
        for index, row in df.iterrows(): 
            mp = row['MPs'] 
            bigram = row['bigrams'] 
            freq = row['freq'] 
            if mp in mp_bigram_count: 
                mp_bigram = mp_bigram_count[mp] 
                if bigram in mp_bigram: 
                    mp_bigram_count[mp][bigram] += freq 
                else: 
                    mp_bigram_count[mp][bigram] = freq 
            else: 
                mp_bigram_count[mp] = {} 
                mp_bigram_count[mp][bigram] = freq

rows = []

for mp in mp_bigram_count.keys():
    for bigram, freq in mp_bigram_count[mp].items():
        rows.append([mp, bigram, freq])
        
df = pd.DataFrame(rows, columns=['MPs', 'bigrams', 'freq'])

df.to_csv('WordFreqYear15-16ByMP_Bigrams.csv')
