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

if True:
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

#%%
import pandas as pd
import config as cf
import numpy as np
import string
import re

testing = False
hier_window = 20
pretrained_model = 'glove-wiki-gigaword-50'

df = pd.read_pickle(cf.B_C_cleaned_data)

# Select only relevant columns and drop NaNs
df = df.loc[:, ['presentation', 'q_and_a', 'price_change']]
df = df.dropna()
docs = pd.concat((df.presentation, df.q_and_a)).reset_index(drop=True)

# Preprocessing
docs = docs.str.lower()

# Remove lower case
docs = docs.str.replace('[{}]'.format(string.punctuation), '')

# Do tokenization and some other stuff
# tokenizer = RegexpTokenizer(r'\w+')

for idx in range(len(docs)):
    docs[idx] = docs[idx].lower()
    docs[idx] = re.sub(r'  ', ' ', docs[idx])
    docs[idx] = re.sub(r'\n', ' ', docs[idx])
    docs[idx] = re.sub(r'â', '', docs[idx])


import gensim.downloader as api
import json

# Print possible pretrained models
info = api.info()
for model_name, model_data in sorted(info['models'].items()):
    print(
        '%s (%d records): %s' % (
            model_name,
            model_data.get('num_records', -1),
            model_data['description'][:40] + '...',
        )
    )

# Load a model
# glove_wiki_50_info = api.info('glove-wiki-gigaword-50')
# print(json.dumps(glove_wiki_50_info, indent=4))
glove_vectors = api.load(pretrained_model)

if testing:
    # Test the woman, king, man question
    result = glove_vectors.most_similar(positive=['woman', 'king'], 
                                        negative=['man'])
    print("Considering the word relationship (king, man) \n The model deems " +\
          f"the pair (woman, ?) to be answered as ? = {result[0][0]} with " +\
          f"score {result[0][1]} ")
    
    # Check similar words
    print(glove_vectors.most_similar("emotional"))
    print(glove_vectors.most_similar("rational"))
    
    # Odd one out
    print(glove_vectors.doesnt_match("breakfast cereal dinner lunch".split()))
    
    # Distance between words
    distance = glove_vectors.distance("coffee", "tea")
    print(f"The distance between the words 'coffee' and 'tea' is: {distance}")
    distance = glove_vectors.distance("coffee", "coffee")
    print("The distance between a word and itself (using w2v) is always: ", 
          distance)

# Add price changes
docs = pd.DataFrame(docs, columns=["Call"])
price_changes = pd.concat((df.price_change, df.price_change))
docs['price_change'] = price_changes.reset_index(drop=True)

# Getting the data into a dataframe
docs['Presentation'] = np.repeat([1, 0], docs.shape[0] / 2)
docs['SWEM-aver'] = 0
docs['SWEM-max'] = 0
docs['SWEM-concat'] = 0
docs['SWEM-hier'] = 0

swem_aver = []
swem_max = []
swem_concat = []
swem_hier = []

for index, row in docs.iterrows():
    call = row["Call"]
    sentence_embedding_list = []
    names = []
    
    missing_words = 0
    for word in call.split():
        try:
            word_vector = glove_vectors[word]
            sentence_embedding_list.append(word_vector)
        except:
            missing_words += 1
            print(f"Word '{word}' is not in the vocabulary.")
    
    if len(sentence_embedding_list) > 0:
        # SWEM-aver
        embedding_aver = np.average(np.asarray(sentence_embedding_list), 
                                    axis=0)
        
        # SWEM-max
        embedding_max = np.max(np.asarray(sentence_embedding_list), axis=0)
        
        # SWEM-concat
        embedding_concat = np.concatenate((embedding_aver, embedding_max))
        
        # SWEM-hier
        hier_window_u = min(hier_window, len(sentence_embedding_list))
        embedding_hier_temp = []
        
        for i in range(len(sentence_embedding_list) - hier_window_u):
            embedding_temp = sentence_embedding_list[i:(i+hier_window_u)]
            embedding_hier_temp.append(np.average(np.asarray(embedding_temp), 
                                                  axis=0))
        
        if len(sentence_embedding_list) == hier_window_u:
            embedding_hier_temp = sentence_embedding_list
        
        embedding_hier = np.max(np.asarray(embedding_hier_temp), axis=0)
    else:
        embedding_aver = np.zeros(glove_vectors['test'].size)
        embedding_max = np.zeros(glove_vectors['test'].size)
        embedding_concat = np.zeros(glove_vectors['test'].size)
        embedding_hier = np.zeros(glove_vectors['test'].size)
    
    swem_aver.append(embedding_aver)
    swem_max.append(embedding_max)
    swem_concat.append(embedding_concat)
    swem_hier.append(embedding_hier)


docs['SWEM-aver'] = swem_aver
docs['SWEM-max'] = swem_max
docs['SWEM-concat'] = swem_concat
docs['SWEM-hier'] = swem_hier












