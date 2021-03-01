import pandas as pd


def test_shares():
    # verify if shares across topics for a month sums up to 1
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    years = [2014, 2015, 2016, 2017, 2018]

    # read the shares dataframe
    df = pd.read_csv('topic_shares_new.csv')

    for year in years:
        for month in months:
            df_month = df.loc[(df.source.str.contains('{}-{}'.format(month, year))) | (df.source.str.contains('{}-{} 54_Others.csv'.format(month, year)))]
            df_month_eu = df.loc[(df.source.str.contains('{}-{} 54_EU.csv'.format(month, year)))]
            df_month_non_eu = df.loc[(df.source.str.contains('{}-{} 54_non_EU.csv'.format(month, year)))]
            df_month_others = df.loc[df.source.str.contains('{}-{} 54_Others.csv'.format(month, year))]

            eu_non_share = df_month_eu['share'].sum() + df_month_non_eu['share'].sum()
            others_share = df_month_others['share'].sum()

            print('eu share {} non_eu share {} total share {}'.format(df_month_eu['share'].sum(), df_month_non_eu['share'].sum(), eu_non_share + others_share))

            total_share = df_month['share'].sum() - eu_non_share
            print('shares for {}-{} are {}'.format(year, month, total_share))


if __name__ == '__main__':
    test_shares()
