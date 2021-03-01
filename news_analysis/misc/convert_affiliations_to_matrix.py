import pandas as pd
import numpy as np
import os

month_name_to_index = {
    'jan': 1,
    'feb': 2,
    'mar': 3,
    'apr': 4,
    'may': 5,
    'jun': 6,
    'jul': 7,
    'aug': 8,
    'sep': 9,
    'oct': 10,
    'nov': 11,
    'dec': 12
}

def split_source_name(source):
    if len(source.split(' ')) == 2 or len(source.split(' ')) == 1:
        i = source.index('_')
        return source[0:i], source[i+1:]
    else:
        split = source.split('_')
        print(split)
        return split[1], split[2]
    
    
def convert_affiliations_to_matrix(dataframe, aff_type, save_name):
    matrix_form = {}
    source_names = []
    for index, row in dataframe.iterrows():
        dt, source_name = split_source_name(row['sources'])
        source_names.append(source_name)
        if dt not in matrix_form:
            matrix_form[dt] = {}
        if source_name not in matrix_form[dt]:
            matrix_form[dt][source_name] = 0
        matrix_form[dt][source_name] = row[aff_type]
    source_names = list(set(source_names))
    source_names.sort()
    
    rows = []
    
    for dt in matrix_form.keys():
        row = []
        row.append(dt)
        month, year = dt.split('-')
        month = month_name_to_index[month]
        
        row.extend([year, month])
        for source in source_names:
            if source in matrix_form[dt]:
                row.append(matrix_form[dt][source])
            else:
                row.append(0)
        rows.append(row)
    columns = ['date', 'year', 'month']
    columns.extend(source_names)
     
    res_df = pd.DataFrame(rows, columns=columns)
    res_df = res_df.sort_values(by=['year', 'month'])
    
    res_df.to_csv(save_name, index=False)
    

df = pd.read_csv('affiliations_std_before.csv')
convert_affiliations_to_matrix(df, 'affiliation_party', 'affiliation_party_std_before.csv')
convert_affiliations_to_matrix(df, 'affiliation_party_EN', 'affiliation_party_EN_std_before.csv')
convert_affiliations_to_matrix(df, 'affiliation_conShare', 'affiliation_conShare_std_before.csv')
convert_affiliations_to_matrix(df, 'affiliation_conShare_EN', 'affiliation_conShare_EN_std_before.csv')
convert_affiliations_to_matrix(df, 'affiliation_conShare2', 'affiliation_conShare2_std_before.csv')
convert_affiliations_to_matrix(df, 'affiliation_conShare2_EN', 'affiliation_conShare2_EN_std_before.csv')

df = pd.read_csv('affiliations_std_after.csv')
convert_affiliations_to_matrix(df, 'affiliation_party', 'affiliation_party_std_after.csv')
convert_affiliations_to_matrix(df, 'affiliation_party_EN', 'affiliation_party_EN_std_after.csv')
convert_affiliations_to_matrix(df, 'affiliation_conShare', 'affiliation_conShare_std_after.csv')
convert_affiliations_to_matrix(df, 'affiliation_conShare_EN', 'affiliation_conShare_EN_std_after.csv')
convert_affiliations_to_matrix(df, 'affiliation_conShare2', 'affiliation_conShare2_std_after.csv')
convert_affiliations_to_matrix(df, 'affiliation_conShare2_EN', 'affiliation_conShare2_EN_std_after.csv')

