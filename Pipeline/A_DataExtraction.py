import os
import re
from datetime import timedelta
from os.path import join

import numpy as np
import pandas as pd
import pandas_datareader as pdr

import config as cf


def simplify(nam):
    # FREE
    return re.sub("[^a-zA-Z]+", "",
                  re.sub('THE ', '', nam.upper()).split(',')[0].split(' ')[0].split('-')[0].split('.')[0])


def transform_to_df():
    # FREE
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

    # Sometimes sentences continue after linebreaks, to get a full sentence on
    # a single line, do the following
    def join_sentences(row):
        if row['call'] is not None:
            row['call'] = re.sub(r'\n,', ' ,', row['call'])
            row['call'] = re.sub(r' ,', ',', row['call'])

            # This can probably be done properly with regex, but I don't know
            # how
            for c in string.ascii_lowercase:
                pattern = f'\\n{c}'
                replace = f' {c}'
                row['call'] = re.sub(pattern, replace, row['call'])

        return row
    df_text = df_text.apply(join_sentences, axis=1)

    # Remove the linebreak at the end of the transcript
    def remove_trailing_lb(row):
        if row['call'] is not None and len(row['call']) > 0:
            if row['call'][-1] == '\n':
                row['call'] = row['call'][:-1]
        return row
    df_text = df_text.apply(remove_trailing_lb, axis=1)

    # Ensure that each line has a single sentence
    def one_sentence_per_line(row):
        if row['call'] is not None and len(row['call']) > 0:
            row['call'] = re.sub(r'([.?!]) ', r'\1\n', row['call'])
            row['call'] = re.sub(r'[.]\n', r'\n', row['call'])
            if row['call'][-1] == '.':
                row['call'] = row['call'][:-1]
        return row
    df_text = df_text.apply(one_sentence_per_line, axis=1)

    print('Finished: Reading the data and returning it as a dataframe')

    return (df_text)


def split_on_qanda(df_text):
    """
    Splits the earnings call into a presentation part and a Q&A part. Splitting
    is done using the word 'question'. If this word does not occur in the text,
    a 2/3 - 1/3 split is made and the final part is considered the Q&A part.
    
    Additional rules can be considered
    
    Parameters
    ----------
    df_text : pandas dataframe
        Dataframe containing at least the transcript of the earnings call in a
        column named 'call'.

    Returns
    -------
    pandas dataframe
        The input dataframe with two additional columns named 'presentation'
        and 'q_and_a'.

    """
    # FREE
    print('Splitting calls into presentation and Q&A parts')
    def get_pres_qanda(row):
        # Select the call
        call_i = row["call"].split('\n')
        
        # Set default value for the Q&A index
        qanda_index = (len(call_i) // 3) * 2
        
        # Adjust the Q&A index based on some keywords
        for j, line in enumerate(call_i):
            if 'question' in line.lower():
                qanda_index = j
                break
            elif 'q&a' in line.lower():
                qanda_index = j
                break
        
        # Create strings for the presentation and Q&A
        presentation = ""
        qanda = ""
        for k, line in enumerate(call_i):
            if k < qanda_index:
                presentation += call_i[k] + '\n'
            else:
                qanda += call_i[k] + '\n'
        
        # Remove trailing linebreaks
        presentation = presentation[:-1]
        qanda = qanda[:-1]
        
        # Return tuple with the presentation and the Q&A
        return presentation, qanda
    
    df_text.assign(presentation="")
    df_text.assign(q_and_a="")
    
    for i, row in df_text.iterrows():
        if row['call'] is not None:
            df_text.loc[i, "presentation"], df_text.loc[i, "q_and_a"] =\
                get_pres_qanda(row)
    
    print('Finished: Splitting calls into presentation and Q&A parts')
    
    return df_text


def get_stock_data(df_text):
    # FREE
    print('Enriching data with prices from the stock market')

    # Request data via Yahoo public API
    def get_comparison_prices(row, which):
        print(f"Getting data for {row['idx']}")
        # Define a wide range of days around the earnings call
        start = row['date'] - timedelta(days=5)
        end = row['date'] + timedelta(days=5)
        
        try:
            # Get the stock prices from yahoo
            prices = pdr.get_data_yahoo(row['idx'], start, end)
        except:
            Warning(f'No values found for company with stock market index {row["idx"]}')
            return np.nan, np.nan
        
        # Get the index of the earning calls date in the dataframe
        date_index = np.where(prices.index == row['date'])[0][0]
        
        # Explanation:
        # prices[whatever you want (e.g. max, min, close)][day number (0 = day before, 1 = day itself, 2 = day after)]
        # Daniel: I noticed that some stock prices already changed on the same
        #         day as the earnings call. So taking the opening price would
        #         miss the price change. Therefore I chose the closing price of
        #         the day before. Because the adjusted closing price is
        #         corrected for events, I think it may be more robust to use
        #         the closing price on the day after the earnings call. If we
        #         do not want to use the adjusted prices, we can take the 
        #         closing price on the day before and the opening price of the
        #         day after.

        # Price before the earnings call
        try:
            price_before = prices["Adj Close"][date_index - 1]  # Adjusted close from the day before
        except IndexError as e:
            price_before = np.nan

        # Price after the earnings call
        try:
            price_after = prices["Adj Close"][date_index + 1]  # Adjusted close from the day after
        except IndexError as e:
            price_after = np.nan

        return price_before, price_after

    print('Requesting stock market prices from the almighty Yahoos. This can take some time!')
    df_text = df_text.assign(price_before=np.nan)
    df_text = df_text.assign(price_after=np.nan)

    # Iterate through the rows of the dataframe to obtain the prices
    for i, row in df_text.iterrows():
        print(f'\rTranscript number: {str(i + 1).zfill(3)}/{df_text.shape[0]}',
              end='\r')
        before, after = get_comparison_prices(row)
        df_text.loc[i, 'price_before'] = before
        df_text.loc[i, 'price_after'] = after
    print("")

    print('Finished: Enriching data with prices from the stock market')
    return df_text


if __name__ == '__main__':
    df_text = transform_to_df()
    df_texts_and_prices = get_stock_data(df_text)
