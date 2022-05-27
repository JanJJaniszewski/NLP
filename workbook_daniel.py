import config as cf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import pandas_datareader as pdr
from datetime import date
from os.path import join

import re
import gensim
import gensim.corpora as corpora

from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

#%%

if False:
    import Pipeline.B_DataRepresentation as B
    import Pipeline.C_LDA as C
    
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Computer Modern Sans Serif']})
    
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex = True)
    
    # B.price_change_summary()
    C.plot_perplexities()

#%%
import Pipeline.C_LDA as C

perplexities = pd.read_pickle(cf.lda_perplexities)
n_topics = perplexities.idxmin()

data = pd.read_pickle(cf.texts_and_prices_file)

# Compute the perplexities for each number of topics using cross validation
res = C.LDA(data, n_topics, default_stopwords=True)

#%%
import Pipeline.B_DataRepresentation as B
B.create_wordcloud()







