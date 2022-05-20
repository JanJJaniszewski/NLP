import re

import config as cf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
from os.path import join

import pandas_datareader as pdr
from datetime import date


def create_wordcloud():
    def cloud(text, filename):
        # Create stopword list:
        stopwords = set(STOPWORDS)
        
        # Add some words to the stop word list, this can be adjusted
        stopwords.update(["yes", "â", "see", "going", "question", "u", "thank", 
                          "you", "itâ", "s", "well", "us", "weâ", "year", 
                          "will", "business", "new", "one", "continue", "end", 
                          "now", "re"])
    
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
    texts = pd.read_pickle(cf.texts_and_prices_file)
    
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
    data = pd.read_pickle(cf.texts_and_prices_file)
    
    if 'price_change' not in data.columns:
        data['price_change'] = data.apply(lambda row: perc_change(row), 1)
    
    # Create histogram
    bins = np.arange(-32.5, 32.5 + 1e-6, 1)
    plt.figure(dpi=400, figsize=[6, 2.2])
    plt.subplot(1,2,1)
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
    positive_mean = data['price_change'][data['price_change'] > 0].mean()
    negative_mean = data['price_change'][data['price_change'] < 0].mean()
    
    # Print information
    print('\nAverage change in stock price is {}{:.2f}%'\
          .format('+' if total_mean >= 0 else '-', np.abs(total_mean) * 100))
    print('Number of positive changes: {} (on average +{:.2f}%)'\
          .format(change_pos, positive_mean * 100))
    print('Number of negative changes: {} (on average -{:.2f}%)'\
          .format(change_neg, np.abs(negative_mean) * 100))
    print(f'Number of no changes: {change_not}')
    print(f'Number of missing values: {missing}\n')
    
    print('Finished generating histogram and summary statistics')
    
    return None


def price_change_summary_2017():
    print('Generating histogram and summary statistics for 2017')
    # Load data
    data = pd.read_pickle(cf.texts_and_prices_file)
    
    # Initialize list for returns
    returns_list = []
    
    # Set start and end date
    start = date(2017, 1, 1)
    end = date(2017, 12, 31)
    
    for i in range(data.shape[0]):
        print(f'\rTranscript number: {str(i + 1).zfill(3)}/{data.shape[0]}',
              end='\r')
        idx_i = data.idx[i]
        
        try:
            # Get the stock prices from yahoo
            prices = pdr.get_data_yahoo(idx_i, start, end)
            prices = prices['Adj Close']
            
            returns = np.zeros(prices.size - 2)
            for i in range(returns.size):
                returns[i] = (prices[i + 2] - prices[i]) / prices[i]
            
            returns_list.append(returns)
        except:
            warning_msg = 'No values found for company with stock market ' +\
                'index {idx_i}'
            Warning(warning_msg)
    
    # Get all returns in a single array
    returns = np.concatenate(returns_list)
    
    # Draw the histogram
    bins = np.arange(-32.5, 32.5 + 1e-6, 1)
    plt.figure(dpi=400, figsize=[6, 2.2])
    plt.subplot(1,2,1)
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
    positive_mean = returns[returns > 0].mean()
    negative_mean = returns[returns < 0].mean()
    
    # Print information
    print('\nAverage change in stock price is {}{:.2f}%'\
          .format('+' if total_mean >= 0 else '-', np.abs(total_mean) * 100))
    print('Number of positive changes: {} (on average +{:.2f}%)'\
          .format(change_pos, positive_mean * 100))
    print('Number of negative changes: {} (on average -{:.2f}%)'\
          .format(change_neg, np.abs(negative_mean) * 100))
    print(f'Number of no changes: {change_not}')
    
    print('Finished generating histogram and summary statistics for 2017')
    
    return None


def transform_to_finbert_format():
    print('Transforming data into Finbert format')
    texts = pd.read_pickle(cf.texts_and_prices_file)
    texts['text'] = texts['presentation']
    texts = texts.dropna()
    texts['label'] = 'neutral'
    texts.loc[((texts['price_after'] - texts['price_before']) / texts['price_before']) > 0.01, 'label'] = 'positive'
    texts.loc[((texts['price_after'] - texts['price_before']) / texts['price_before']) < -0.01, 'label'] = 'negative'
    for_finbert = texts[['text', 'label']]
    train, validate, test = \
        np.split(for_finbert.sample(frac=1, random_state=42),
                 [int(.6 * len(for_finbert)), int(.8 * len(for_finbert))])
    train.to_csv(cf.path_train, sep="\t")
    test.to_csv(cf.path_test, sep="\t")
    validate.to_csv(cf.path_validate, sep="\t")
    print(f'Saved all training, test, and validation in Finbert format to {cf.path_train}, {cf.path_validate}, {cf.path_test}')


if __name__ == '__main__':
    transform_to_finbert_format()