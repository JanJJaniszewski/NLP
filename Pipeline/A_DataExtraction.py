import os
import re
from datetime import timedelta
from os.path import join

import numpy as np
import pandas as pd
import pandas_datareader as pdr

import config as cf


def simplify(nam):
    return re.sub("[^a-zA-Z]+", "",
                  re.sub('THE ', '', nam.upper()).split(',')[0].split(' ')[0].split('-')[0].split('.')[0])


def transform_to_df():
    from config_dictionaries import company_name_mapping

    print('Reading the data and returning it as a dataframe')
    folders = os.listdir(cf.Z_A_zipfolder)

    # Get metadata on calls
    company_names = [name.split('_')[0] for name in folders]
    company_name_mapping = {simplify(nam): sh for nam, sh in company_name_mapping.items()}
    company_idx = [company_name_mapping[simplify(nam)] for nam in company_names]

    call_dates = pd.to_datetime([name.split('_')[1].split('.')[0] for name in folders])

    # Get texts themselves
    def read_file(filename):
        with open(filename) as file:
            try:
                return file.read()
            except UnicodeDecodeError as ude:
                Exception(f"{filename}: {ude}")

    texts = [read_file(join(cf.Z_A_zipfolder, filename)) for filename in folders]
    df_text = pd.DataFrame({'name': company_names, 'idx': company_idx, 'date': call_dates, 'call': texts})
    print('Finished: Reading the data and returning it as a dataframe')

    return (df_text)


def get_stock_data(df_text):
    print('Enriching data with prices from the stock market')
    # Request data via Yahoo public API
    def get_comparison_prices(row, which):
        print('.')
        try:
            prices = pdr.get_data_yahoo(row['idx'], row['date'] - timedelta(days=1), row['date'] + timedelta(days=1))
        except:
            Warning(f'No values found for company with stock market index {row["idx"]}')
            return np.nan

        # Explanation:
        # prices[whatever you want (e.g. max, min, close)][day number (0 = day before, 1 = day itself, 2 = day after)]
        if which == 'before':
            return prices['Close'][1]  # Open price from day itself
        elif which == 'after':
            return prices['Open'][2]  # Open price from day after

    print('Requesting stock market prices from the almighty Yahoos. This can take some time!')
    df_text['price_before'] = df_text.apply(lambda row: get_comparison_prices(row, 'before'), 1)
    df_text['price_after'] = df_text.apply(lambda row: get_comparison_prices(row, 'after'), 1)

    # Save dataset just in case
    print(f'Saving dataset in {cf.texts_and_prices_file}')
    df_text.to_pickle(cf.texts_and_prices_file)

    print('Finished: Enriching data with prices from the stock market')
    return df_text


if __name__ == '__main__':
    df_text = transform_to_df()
    df_texts_and_prices = get_stock_data(df_text)
