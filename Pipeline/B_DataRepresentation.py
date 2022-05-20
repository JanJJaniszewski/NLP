import config as cf
import pandas as pd
import numpy as np
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
    df['presentation'] = [remove_stopwords(t) for t in df['presentation']]
    df['q_and_a'] = [remove_stopwords(t) for t in df['q_and_a']]

    print('Finished dropping stopwords')
    return df

def transform_to_finbert_format():
    print('Transforming data into Finbert format')
    texts = pd.read_pickle(cf.A_B_texts_and_prices_file)
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