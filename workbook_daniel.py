import config as cf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def perc_change(row):
    return (row['price_after'] - row['price_before']) / row['price_before']


if __name__ == '__main__':
    data = pd.read_pickle(cf.texts_and_prices_file)
    data['change'] = data.apply(lambda row: perc_change(row), 1)
    
    plt.figure(dpi=300)
    plt.hist(data['change'] * 100, bins=40)
    plt.xlim([-0.32 * 100, 0.32 * 100])
    plt.xlabel('Change in stock price (in %)')
    plt.ylabel('Frequency')
    plt.show()
    
    change_pos = sum(data['change'] > 0)
    change_neg = sum(data['change'] < 0)
    change_not = sum(data['change'] == 0)
    missing = data.shape[0] - change_pos - change_neg - change_not
    
    total_mean = data['change'].mean()
    positive_mean = data['change'][data['change'] > 0].mean()
    negative_mean = data['change'][data['change'] < 0].mean()
    
    print('Average change in stock price is {}{:.2f}%'\
          .format('+' if total_mean >= 0 else '-', np.abs(total_mean) * 100))
    print('Number of positive changes: {} (on average +{:.2f}%)'\
          .format(change_pos, positive_mean * 100))
    print('Number of negative changes: {} (on average -{:.2f}%)'\
          .format(change_neg, np.abs(negative_mean) * 100))
    print(f'Number of no changes: {change_not}')
    print(f'Number of missing values: {missing}')
