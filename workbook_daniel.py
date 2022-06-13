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
from tabulate import tabulate
import pandas as pd
import config as cf
import numpy as np
import string
import re
import gensim.downloader as api
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

def swem_preprocess(df, pretrained_model, hier_window, pca_components, 
                    testing=False, verbose=0):
    # Select only relevant columns and drop NaNs
    df = df.loc[:, ['presentation', 'q_and_a', 'price_change']]
    df = df.dropna()
    docs = pd.concat((df.presentation, df.q_and_a)).reset_index(drop=True)
    
    # Preprocessing
    docs = docs.str.lower()
    
    # Remove lower case
    docs = docs.str.replace('[{}]'.format(string.punctuation), '', regex=True)
    
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()
        docs[idx] = re.sub(r'  ', ' ', docs[idx])
        docs[idx] = re.sub(r'\n', ' ', docs[idx])
        docs[idx] = re.sub(r'â', '', docs[idx])
    
    # Print possible pretrained models
    if verbose > 0:
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
        print("Considering the word relationship (king, man) \n The model " +\
              "deems the pair (woman, ?) to be answered as ? = " +\
              f"{result[0][0]} with score {result[0][1]} ")
        
        # Check similar words
        print(glove_vectors.most_similar("emotional"))
        print(glove_vectors.most_similar("rational"))
        
        # Odd one out
        print(glove_vectors.doesnt_match(
            "breakfast cereal dinner lunch".split()))
        
        # Distance between words
        distance = glove_vectors.distance("coffee", "tea")
        print("The distance between the words 'coffee' and 'tea' is:" +\
              f" {distance}")
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
    docs['SWEM-min'] = 0
    docs['SWEM-concat'] = 0
    docs['SWEM-hier'] = 0
    docs['SWEM-pca'] = 0
    
    swem_aver = []
    swem_max = []
    swem_min = []
    swem_concat = []
    swem_hier = []
    
    for index, row in docs.iterrows():
        call = row["Call"]
        sentence_embedding_list = []
        
        missing_words = 0
        for word in call.split():
            try:
                word_vector = glove_vectors[word]
                sentence_embedding_list.append(word_vector)
            except:
                missing_words += 1
                # print(f"Word '{word}' is not in the vocabulary.")
        
        if len(sentence_embedding_list) > 0:
            # SWEM-aver
            embedding_aver = np.average(np.asarray(sentence_embedding_list), 
                                        axis=0)
            
            # SWEM-max
            embedding_max = np.max(np.asarray(sentence_embedding_list), axis=0)
            
            # SWEM-min
            embedding_min = np.min(np.asarray(sentence_embedding_list), axis=0)
            
            # SWEM-concat
            embedding_concat = np.concatenate((embedding_aver, embedding_max))
            
            # SWEM-hier
            hier_window_u = min(hier_window, len(sentence_embedding_list))
            embedding_hier_temp = []
            
            for i in range(len(sentence_embedding_list) - hier_window_u):
                embedding_temp = sentence_embedding_list[i:(i+hier_window_u)]
                embedding_hier_temp.append(
                    np.average(np.asarray(embedding_temp), axis=0))
            
            if len(sentence_embedding_list) == hier_window_u:
                embedding_hier_temp = sentence_embedding_list
            
            embedding_hier = np.max(np.asarray(embedding_hier_temp), axis=0)
        else:
            embedding_aver = np.zeros(glove_vectors['test'].size)
            embedding_max = np.zeros(glove_vectors['test'].size)
            embedding_min = np.zeros(glove_vectors['test'].size)
            embedding_concat = np.zeros(glove_vectors['test'].size * 2)
            embedding_hier = np.zeros(glove_vectors['test'].size)
        
        swem_aver.append(embedding_aver)
        swem_max.append(embedding_max)
        swem_min.append(embedding_min)
        swem_concat.append(embedding_concat)
        swem_hier.append(embedding_hier)
    
    
    docs['SWEM-aver'] = swem_aver
    docs['SWEM-min'] = swem_min
    docs['SWEM-max'] = swem_max
    docs['SWEM-concat'] = swem_concat
    docs['SWEM-hier'] = swem_hier
    
    # Compute SWEM-pca
    M = np.array(list(docs['SWEM-aver']))
    pca = PCA(n_components=2)
    docs['SWEM-pca'] = list(pca.fit_transform(M))
    
    return docs


def neural_network_accuracy(data, method, k):
    def get_Xy(df, method):
        array_X = []
        array_y = []
        for index, row in df.iterrows():
            array_X.append(row[method])
            array_y.append(row['outcome'])
        array_X = np.array(array_X)
        array_y = np.array(array_y)
        
        return array_X, array_y
    
    n = data.shape[0]
    accuracies = []
    f1s = []
    
    for k_i in range(k):
        idx0, idx1 = int(k_i * n / 5), int((k_i + 1) * n / 5)
        test_idx = list(range(idx0, idx1))
        
        train = data.drop(test_idx).reset_index(drop=True)
        test = data.loc[test_idx, :]
        
        train_X, train_y = get_Xy(train, method)
        test_X, test_y = get_Xy(test, method)
        
        # Define model
        model = Sequential()
        model.add(Dense(train_X.shape[1] * 2, input_dim=train_X.shape[1], 
                        activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(train_X.shape[1] * 2, activation='relu', 
                        kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(train_y.shape[1], activation='softmax'))
        
        # Compile
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        # Fit
        model.fit(train_X, train_y, epochs=100, batch_size=int(n / 5), 
                  verbose=0)
        
        # Compute accuracy
        y_pred = model.predict(test_X, verbose=False)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(test_y, axis=1)
        
        # print(y_pred)
        accuracies.append(sum(y_true == y_pred))
        f1s.append(f1_score(y_true, y_pred, average='macro') * y_true.size)
    
    accuracy = sum(accuracies) / n
    f1 = sum(f1s) / n
    
    return accuracy, f1


def encode_price_change(row, m, s):
    price_change = row['price_change']
    
    if price_change < m - s:
        return np.array([1, 0, 0])
    elif price_change >= m - s and price_change <= m + s:
        return np.array([0, 1, 0])
    elif price_change > m + s:
        return np.array([0, 0, 1])


def SWEM_analysis():
    texts = pd.read_pickle(cf.B_C_cleaned_data).sample(frac=1)\
        .reset_index(drop=True)
    pretrained_models = ['glove-twitter-50', 'glove-wiki-gigaword-50', 
                         'glove-wiki-gigaword-200', 'glove-twitter-200',
                         'glove-wiki-gigaword-300', 'word2vec-google-news-300']
    results_dict = {}
    
    for pretrained_model in pretrained_models:
        print(f'\nUsing {pretrained_model} as pretrained model')
        # Preprocess the data
        texts_processed = swem_preprocess(texts, 
            pretrained_model=pretrained_model, hier_window=20, 
            pca_components=50)
        
        # Make the outcome discrete
        texts_processed['outcome'] = texts_processed.apply(encode_price_change, 
            axis=1, args=(0.0015, 0.0201))
        
        # Divide the data into two parts
        n = int(texts_processed.shape[0] / 2)
        
        # Separate into two data sets (presentation and Q&A)
        data_pr = texts_processed.iloc[:n, :].reset_index(drop=True)
        data_qa = texts_processed.iloc[n:, :].reset_index(drop=True)
        
        # Compute results
        results = pd.DataFrame(0, columns=['Pres-acc', 'Pres-f1', 'Q&A-acc', 
                                           'Q&A-f1'],
                               index=['SWEM-aver', 'SWEM-max', 'SWEM-min', 
                                      'SWEM-concat', 'SWEM-hier', 'SWEM-pca'])
        
        n_repeats = 5
        k = 5
        
        print('Training neural networks on presentation data...')
        for method in results.index:
            for i in range(n_repeats):
                scores = neural_network_accuracy(data_pr, method, k)
                results.loc[method, 'Pres-acc'] += scores[0] / n_repeats
                results.loc[method, 'Pres-f1'] += scores[1] / n_repeats
        # print(results)
        
        print('Training neural networks on Q&A data...')
        for method in results.index:
            for i in range(n_repeats):
                scores = neural_network_accuracy(data_qa, method, k)
                results.loc[method, 'Q&A-acc'] += scores[0] / n_repeats
                results.loc[method, 'Q&A-f1'] += scores[1] / n_repeats
        
        print(tabulate(results, headers=results.columns, tablefmt="fancy_grid",
                       floatfmt=".3f"))
    
        results_dict[pretrained_model] = results
    
    return results_dict

#%%

results_dict = SWEM_analysis()

for key in results_dict:
    print(key)
    print(tabulate(results_dict[key], headers=results_dict[0].columns,
                   tablefmt="latex", floatfmt=".3f"))
    print('\n\n')

A = results_dict['word2vec-google-news-300']
B = results_dict['glove-wiki-gigaword-300']


for i in range(5):
    row = ''
    row += '{:.3f} & {:.3f} && '.format(A.iloc[i, 0], A.iloc[i, 1])
    row += '{:.3f} & {:.3f} && '.format(A.iloc[i, 2], A.iloc[i, 3])
    row += '{:.3f} & {:.3f} && '.format(B.iloc[i, 0], B.iloc[i, 1])
    row += '{:.3f} & {:.3f} \\\\'.format(B.iloc[i, 2], B.iloc[i, 3])
    print(row)
    




