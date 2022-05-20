import config as cf
import pandas as pd
import matplotlib.pyplot as plt
import gensim.corpora as corpora
import numpy as np

import re
import gensim
import nltk

from gensim.models import Phrases
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from os.path import join



def LDA_preprocess2(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    
    for idx in range(len(docs)):
        docs[idx] = docs[idx].lower()
        docs[idx] = re.sub(r'\n', ' ', docs[idx])
        docs[idx] = re.sub(r'â', '', docs[idx]) # Check whether there is a hat on the a
        docs[idx] = tokenizer.tokenize(docs[idx])
    
    # Remove numbers, but not words that contain numbers.
    docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
    
    # Remove words that are only one character.
    docs = [[token for token in doc if len(token) > 1] for doc in docs]
    
    # Lemmatize
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()
    docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
    
    # Bigrams
    bigram = Phrases(docs, min_count=20)
    for idx in range(len(docs)):
        for token in bigram[docs[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                docs[idx].append(token)
    
    # Create a dictionary representation of the documents.
    dictionary = corpora.Dictionary(docs)
    
    # Filter out words that occur less than 20 documents, or more than 50% of 
    # the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.5)
    
    # Bag-of-words representation of the documents.
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))
    
    # Make an index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token
    
    return corpus, id2word    


def LDA_perplexities(n_topics, passes=50, k=4, n_repeats=3, 
                     save_perplexities=True):
    def LDA_CV(corpus, id2word, num_topics, passes, k, n_repeats):
        # Perplexity, but also try coherence
        perplexity = 0
        
        for i in range(k):
            # Start and end indices of the test fold
            start = i * round(len(corpus) / k + 0.5)
            end = min((i + 1) * round(len(corpus) / k + 0.5), len(corpus))
            
            # Keep track of the perplexity of this fold
            perplexity_i = 0
            
            for j in range(n_repeats):
                message = f'Number of topics: {str(num_topics).zfill(2)}, ' +\
                    f'fold {i + 1}/{k}, pass {j + 1}/{n_repeats}'
                print(f'\r{message}', end='\r')
                
                # Build LDA model
                lda_model = gensim.models.LdaModel(corpus=corpus[:start] +\
                                                   corpus[end:],
                                                   id2word=id2word,
                                                   chunksize=2000,
                                                   alpha='auto',
                                                   eta='auto',
                                                   iterations=400,
                                                   num_topics=num_topics,
                                                   passes=passes,
                                                   eval_every=None)
                
                # Add to perplexity_i
                perplexity_i += 2**(-lda_model.log_perplexity(\
                    corpus[start:end], total_docs=end - start))
            
            # Add to perplexity
            perplexity += perplexity_i / n_repeats
        
        # Compute average perplexity based on cross validation
        perplexity /= k
        
        print('\n\tAverage perplexity: {:.2f}'.format(perplexity))
        
        return perplexity
    
    print('Computing LDA perplexities')
    
    # Load data
    data = pd.read_pickle(cf.texts_and_prices_file).sample(frac=1)
    
    # Select only relevant columns and drop NaNs
    data = data.loc[:, ['presentation', 'q_and_a']]
    data = data.dropna()
    docs = pd.concat((data.presentation, data.q_and_a)).reset_index(drop=True)
    
    # Apply preprocessing
    corpus, id2word = LDA_preprocess2(docs)
    
    # Initialize perplexity scores
    perplexities = np.zeros(len(n_topics))
    
    # Compute the perplexities for each number of topics using cross validation
    for i, n in enumerate(n_topics):
        perplexities[i] = LDA_CV(corpus, id2word, n, passes, k, n_repeats)
    print('Finished computing LDA perplexities')
    
    perplexities = pd.Series(perplexities, index=n_topics)
    
    if save_perplexities:
        perplexities.to_pickle(cf.lda_perplexities)
    
    return perplexities


def LDA(num_topics=None, passes=50):
    # Load data
    data = pd.read_pickle(cf.texts_and_prices_file).sample(frac=1)
    
    # Select only relevant columns and drop NaNs
    data = data.loc[:, ['presentation', 'q_and_a']]
    data = data.dropna()
    docs = pd.concat((data.presentation, data.q_and_a)).reset_index(drop=True)
    
    # Apply preprocessing
    corpus, id2word = LDA_preprocess2(docs)
    
    # Get the number of topics
    if num_topics is None:
        perplexities = pd.read_pickle(cf.lda_perplexities)
        num_topics = perplexities.idxmin()
    
    # Build LDA model
    lda_model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=id2word,
                                       chunksize=2000,
                                       alpha='auto',
                                       eta='auto',
                                       iterations=400,
                                       num_topics=num_topics,
                                       passes=passes,
                                       eval_every=None)
    
    # Transform topics into dataframe
    lda_topics = lda_model.print_topics()
    str_topics = []
    for t in lda_topics:
        str_topics.append(re.findall(r'"(.*?)"', t[1]))
    colnames = [f'topic {i+1}' for i in range(len(str_topics))]
    lda_topics = pd.DataFrame(str_topics, index=colnames).T
    
    print('These are the topics that are found:')
    print(lda_topics)
    
    # Save output
    lda_topics.to_pickle(cf.lda_topics)
    
    # Save model to disk.
    lda_model.save(cf.lda_model)
    
    # Load a potentially pretrained model from disk.
    # lda = LdaModel.load(cf.lda_model)
    
    return None


def plot_perplexities():
    perplexities = pd.read_pickle(cf.lda_perplexities)
    plt.figure(dpi=400, figsize=[10, 2])
    plt.subplot(1,2,1)
    plt.plot(perplexities, linewidth=0.75, c="black", ls="--")
    plt.xticks([3, 6, 9, 12, 15, 18], fontsize=8)
    plt.yticks([275, 325, 375, 425], fontsize=8)
    plt.ylim(250, 450)
    plt.xlabel(r'Numer of topics ($k$)')
    plt.ylabel('Perplexity')
    plt.savefig(join(cf.path_images, 'LDA_Perplexities.pdf'), 
                bbox_inches="tight")
    return None
