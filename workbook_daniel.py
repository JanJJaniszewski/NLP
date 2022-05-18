import config as cf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas_datareader as pdr
from datetime import date
from os.path import join

#%%

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
    bins = np.arange(-32, 32+1e-6, 1)
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

#%%

import Pipeline.B_DataRepresentation as B

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Sans Serif"]})

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex = True)


B.price_change_summary()












