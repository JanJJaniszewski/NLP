import re
from datetime import date
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from wordcloud import WordCloud, STOPWORDS

import config as cf


def create_wordcloud():
    def cloud(text, filename):
        # Create stopword list:
        stopwords = set(STOPWORDS)

        # Add some words to the stop word list, this can be adjusted
        stopwords.update(["yes", "â", "see", "going", "question", "u", "thank", ',',
                          "you", "itâ", "s", "well", "us", "weâ", "year",
                          "will", "business", "new", "one", "continue", "end",
                          "now", "re", 'thank you', 'thanks', 'Thank', 'Thanks', 'good morning', 'morning'])

        # Generate a word cloud image
        wordcloud = WordCloud(width=600, height=400, max_font_size=90,
                              max_words=50, stopwords=stopwords,
                              background_color="white").generate(text)

        # Display the generated image
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

        # Save the wordcloud
        wordcloud.to_file(join(cf.path_images, filename))

        return None

    print('Generating wordclouds for presentations and Q&A sessions')

    # Load data
    texts = pd.read_pickle(cf.A_B_texts_and_prices_file)

    # Create word cloud for presentations
    text = ""
    for t in texts.presentation:
        if str(t) != 'nan':
            text += ' ' + t
    text = re.sub('\n', ' ', text)
    cloud(text, "wordcloud_presentations.png")

    # Create word cloud for Q&A sessions
    text = ""
    for t in texts.q_and_a:
        if str(t) != 'nan':
            text += ' ' + t
    text = re.sub('\n', ' ', text)
    cloud(text, "wordcloud_Q&A.png")

    print('Finished generating wordclouds for presentations and Q&A sessions')

    return None


def price_change_summary():
    def perc_change(row):
        return (row['price_after'] - row['price_before']) / row['price_before']

    print('Generating histogram and summary statistics')

    # Load data
    data = pd.read_pickle(cf.A_B_texts_and_prices_file)

    if 'price_change' not in data.columns:
        data['price_change'] = data.apply(lambda row: perc_change(row), 1)

    # Create histogram
    bins = np.arange(-32.5, 32.5 + 1e-6, 1)
    plt.figure(dpi=400, figsize=[6, 2.2])
    plt.subplot(1, 2, 1)
    plt.hist(data['price_change'] * 100, bins=bins, density=True, rwidth=0.6)
    plt.xlim([-0.32 * 100, 0.32 * 100])
    plt.ylim([0, 0.32])
    plt.xlabel(r'Two-day returns (in \%)')
    plt.ylabel('Frequency')
    plt.xticks([-25, -12.5, 0, 12.5, 25], fontsize=8)
    plt.yticks([0.0, 0.075, 0.15, 0.225, 0.3], fontsize=8)
    plt.savefig(join(cf.path_images, 'price_change_hist.pdf'),
                bbox_inches="tight")

    # Count the number of positive and negative changes in price
    change_pos = sum(data['price_change'] > 0)
    change_neg = sum(data['price_change'] < 0)
    change_not = sum(data['price_change'] == 0)
    missing = data.shape[0] - change_pos - change_neg - change_not

    # Compute the means
    total_mean = data['price_change'].mean()
    total_std = data['price_change'].std()
    positive_mean = data['price_change'][data['price_change'] > 0].mean()
    positive_std = data['price_change'][data['price_change'] > 0].std()
    negative_mean = data['price_change'][data['price_change'] < 0].mean()
    negative_std = data['price_change'][data['price_change'] < 0].std()

    # Print information
    print('\nAverage change in stock price is {}{:.2f}%' \
          .format('+' if total_mean >= 0 else '-', np.abs(total_mean) * 100))
    print('Standard deviation of the change in stock price is {:.2f}%' \
          .format(np.abs(total_std) * 100))
    print('Number of positive changes: {} (avg: +{:.2f}%, std: {:.2f}%)' \
          .format(change_pos, positive_mean * 100, positive_std * 100))
    print('Number of negative changes: {} (avg: -{:.2f}%, std: {:.2f}%)' \
          .format(change_neg, np.abs(negative_mean) * 100, negative_std * 100))
    print(f'Number of no changes: {change_not}')
    print(f'Number of missing values: {missing}\n')

    print('Finished generating histogram and summary statistics')

    return None


def price_change_summary_2017():
    print('Generating histogram and summary statistics for 2017')
    # Load data
    data = pd.read_pickle(cf.A_B_texts_and_prices_file)

    # Initialize list for returns
    returns_list = []

    # Set start and end date
    start = date(2017, 1, 1)
    end = date(2017, 12, 31)
    
    tickers = list(set(data['idx']))

    for j in range(len(tickers)):
        print(f'\rTranscript number: {str(j + 1).zfill(3)}/{len(tickers)}',
              end='\r')
        idx_i = data.idx[j]

        try:
            # Get the stock prices from yahoo
            prices = pdr.get_data_yahoo(idx_i, start, end)
            prices = prices['Adj Close']

            returns = np.zeros(prices.size - 2)
            for i in range(returns.size):
                returns[i] = (prices[i + 2] - prices[i]) / prices[i]

            returns_list.append(returns)
        except:
            warning_msg = 'No values found for company with stock market ' + \
                          'index {idx_i}'
            Warning(warning_msg)

    # Get all returns in a single array
    returns = np.concatenate(returns_list)

    # Draw the histogram
    bins = np.arange(-32.5, 32.5 + 1e-6, 1)
    plt.figure(dpi=400, figsize=[6, 2.2])
    plt.subplot(1, 2, 1)
    plt.hist(returns * 100, bins=bins, density=True, rwidth=0.6)
    plt.xlim([-0.32 * 100, 0.32 * 100])
    plt.ylim([0, 0.32])
    plt.xlabel(r'Two-day returns (in \%)')
    plt.ylabel('Frequency')
    plt.xticks([-25, -12.5, 0, 12.5, 25], fontsize=8)
    plt.yticks([0.0, 0.075, 0.15, 0.225, 0.3], fontsize=8)
    plt.savefig(join(cf.path_images, 'price_change_hist_2017.pdf'),
                bbox_inches="tight")

    # Count the number of positive and negative changes in price
    change_pos = sum(returns > 0)
    change_neg = sum(returns < 0)
    change_not = sum(returns == 0)

    # Compute the means
    total_mean = returns.mean()
    total_std = returns.std()
    positive_mean = returns[returns > 0].mean()
    positive_std = returns[returns > 0].std()
    negative_mean = returns[returns < 0].mean()
    negative_std = returns[returns < 0].std()

    # Print information
    print('\nAverage change in stock price is {}{:.2f}%' \
          .format('+' if total_mean >= 0 else '-', np.abs(total_mean) * 100))
    print('Standard deviation of the change in stock price is {:.2f}%' \
          .format(np.abs(total_std) * 100))
    print('Number of positive changes: {} (avg: +{:.2f}%, std: {:.2f}%)' \
          .format(change_pos, positive_mean * 100, positive_std * 100))
    print('Number of negative changes: {} (avg: -{:.2f}%, std: {:.2f}%)' \
          .format(change_neg, np.abs(negative_mean) * 100, negative_std * 100))
    print(f'Number of no changes: {change_not}')

    print('Finished generating histogram and summary statistics for 2017')

    return None


from gensim.parsing.preprocessing import remove_stopwords
import urllib.request


def names_drop(df):
    print('Dropping names')
    names = urllib.request.urlopen(
        'https://www.usna.edu/Users/cs/roche/courses/s15si335/proj1/files.php%3Ff=names.txt&downloadcode=yes')
    names = str(names.read()).split('\\n')
    for n in names:
        df['presentation'] = df['presentation'].str.replace(n, '')
        df['q_and_a'] = df['q_and_a'].str.replace(n, '')

    print('Finished dropping names')
    return df


def stopwords_drop(df):
    print('Dropping stopwords')
    df['presentation'] = df['presentation'].apply(lambda row:
                                                  row if isinstance(row, str) else '')
    df['q_and_a'] = df['q_and_a'].apply(lambda row:
                                        row if isinstance(row, str) else '')
    df['presentation'] = [remove_stopwords(t) for t in df['presentation']]
    df['q_and_a'] = [remove_stopwords(t) for t in df['q_and_a']]

    print('Finished dropping stopwords')
    return df


def change_number_to_word(word, debug=False):
    toadd = ''
    word = word.replace(',', '.')
    if re.findall('\d', word) == []:
        return word
    if '$' in word:
        word = word.replace('$', '')
        toadd = '$'
    if '%' in word:
        word = word.replace('%', '')
        toadd = '%'
        try:
            perc = float(word)
        except ValueError:
            pass
        else:
            if perc < 0:
                word = 'negative'
            elif perc < 0.2:
                word = 'low'
            elif perc < 0.4:
                word = 'midlow'
            elif perc < 0.6:
                word = 'midhigh'
            elif perc < 0.8:
                word = 'high'
            else:
                word = 'veryhigh'
    else:
        try:
            # round number
            num = float(word)
        except ValueError:
            pass
        else:
            divisor = int('1' + ((len(word.split('.')[0])-1) * '0'))
            num /= divisor
            num = round(num)
            num *= divisor
            word = str(num)

    word += toadd
    return word

def change_numbers_in_row(row):
    row = row.split(' ')
    row = [change_number_to_word(word) for word in row]
    row = ' '.join(row)


    return row

def change_numbers(df):
    print('Changing numbers')
    df['presentation'] = df['presentation'].apply(lambda row: change_numbers_in_row(row))
    df['q_and_a'] = df['q_and_a'].apply(lambda row: change_numbers_in_row(row))
    print('Finished changing numbers')

    return df


def transform_to_finbert_format(texts, column_to_transform='presentation'):
    print('Transforming data into Finbert format')
    texts['text'] = texts[column_to_transform]
    texts = texts.dropna()
    texts['label'] = 'neutral'
    texts.loc[((texts['price_after'] - texts['price_before']) / texts['price_before']) > 0.0016 + 0.02, 'label'] = 'positive'
    texts.loc[((texts['price_after'] - texts['price_before']) / texts['price_before']) < 0.0016 - 0.02, 'label'] = 'negative'
    for_finbert = texts[['text', 'label']]
    train, validate, test = \
        np.split(for_finbert.sample(frac=1, random_state=42),
                 [int(.6 * len(for_finbert)), int(.8 * len(for_finbert))])
    train.to_csv(cf.path_train, sep="\t")
    test.to_csv(cf.path_test, sep="\t")
    validate.to_csv(cf.path_validate, sep="\t")
    print(
        f'Saved all training, test, and validation in Finbert format to {cf.path_train}, {cf.path_validate}, {cf.path_test}')

    return pd.DataFrame()


if __name__ == '__main__':
    transform_to_finbert_format()


