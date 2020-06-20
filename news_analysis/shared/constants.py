
# months short name used for naming news articles csv files
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

# names of the news sources
source_names = ['Sun', 'Mirror', 'Belfast Telegraph', 'Record', 'Independent', 'Observer', 'Guardian', 'People',
                'Telegraph', 'Mail', 'Express', 'Post', 'Herald', 'Star', 'Wales', 'Scotland', 'Standard', 'Scotsman']

# ids for each source
source_ids = ['400553', '377101', '418973', '244365', '8200', '412338', '138794', '232241', '334988', '331369',
              '138620', '419001', '8010', '142728', '408506', '143296', '363952', '145251', '232240', '145253',
              '389195', '145254', '344305', '8109', '397135', '163795', '412334', '408508', '411938']

# different median clusters possible
median_clusters_name = [
    'overall_in_tomorrows_cluster_above_median',
    'overall_in_tomorrows_cluster_below_median',
    'overall_in_cluster_above_median',
    'overall_in_cluster_below_median',
    'source_in_tomorrows_cluster_above_median',
    'source_in_tomorrows_cluster_below_median',
    'source_in_cluster_above_median',
    'source_in_cluster_below_median'
]

# map of news source id to news source name
id_to_name_map = {
    '400553': 'Belfast Telegraph',
    '377101': 'Scotsman',
    '418973': 'Record',
    '244365': 'Wales',
    '8200': 'Independent',
    '412338': 'Wales',
    '138794': 'Mail',
    '232241': 'Express',
    '334988': 'Telegraph',
    '331369': 'Sun',
    '138620': 'Guardian',
    '419001': 'Mirror',
    '8010': 'Guardian',
    '142728': 'Herald',
    '408506': 'Express',
    '143296': 'Observer',
    '363952': 'Star',
    '145251': 'People',
    '232240': 'Express',
    '145253': 'Record',
    '389195': 'Telegraph',
    '145254': 'Mirror',
    '344305': 'Scotland',
    '8109': 'Telegraph',
    '397135': 'Mail',
    '163795': 'Belfast Telegraph',
    '412334': 'Post',
    '408508': 'Star',
    '411938': 'Standard'
}

# topics id
topics_id = ['02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18',
             '19', '20', '21']

# map for topics id to topics name
topics_id_to_name_map = {
    '02': 'Agriculture, animals, food and rural affairs',
    '03': 'Asylum, immigration and nationality',
    '04': 'Business, industry and consumers',
    '05': 'Communities and families',
    '06': 'Crime, civil law, justice and rights',
    '07': 'Culture, media and sport',
    '08': 'Defence',
    '09': 'Economy and finance',
    '10': 'Education',
    '11': 'Employment and training',
    '12': 'Energy and environment',
    '13': 'European Union',
    '14': 'Health services and medicine',
    '15': 'Housing and planning',
    '16': 'International affairs',
    '17': 'Parliament, government and politics',
    '18': 'Science and technology',
    '19': 'Social security and pensions',
    '20': 'Social services',
    '21': 'Transport'
}

# map for topics index to topics name
topics_index_to_name_map = {
    0: 'Agriculture, animals, food and rural affairs',
    1: 'Asylum, immigration and nationality',
    2: 'Business, industry and consumers',
    3: 'Communities and families',
    4: 'Crime, civil law, justice and rights',
    5: 'Culture, media and sport',
    6: 'Defence',
    7: 'Economy and finance',
    8: 'Education',
    9: 'Employment and training',
    10: 'Energy and environment',
    11: 'European Union',
    12: 'Health services and medicine',
    13: 'Housing and planning',
    14: 'International affairs',
    15: 'Parliament, government and politics',
    16: 'Science and technology',
    17: 'Social security and pensions',
    18: 'Social services',
    19: 'Transport',
    20: 'Others'
}
