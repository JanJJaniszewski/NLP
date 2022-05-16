import config as cf
import pandas as pd
import numpy as np

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