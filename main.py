import Pipeline.A_DataExtraction as A
import Pipeline.B_DataRepresentation as B
import Pipeline.C_LDA as C
import Pipeline.D_TextStructure as D
import Pipeline.E_AttentionModel as E
import Pipeline.F_BERT as F
import Pipeline.Z_DataUnpacking as Z
import config as cf

import pandas as pd

def execute_pipeline(parts):
    if 'Z' in parts:
        Z.unpack_data()
        Z.rename_data()
    if 'A' in parts:
        # Loading (in transform_to_df) and Executing
        df_text = A.transform_to_df()
        df_text = A.split_on_qanda(df_text)
        df_texts_and_prices = A.get_stock_data(df_text)

        # Saving
        df_texts_and_prices.to_pickle(cf.A_B_texts_and_prices_file)

    if 'B' in parts:
        # Set to false if figures shouldn't be redrawn,
        # B.price_change_summary_2017() takes quite a while
        if True:
            B.create_wordcloud()
            B.price_change_summary()
            B.price_change_summary_2017()
        df_text = B.transform_to_finbert_format()

        # Loading
        texts = pd.read_pickle(cf.A_B_texts_and_prices_file)
        # Executing
        if 'B_names' in parts:
            texts = B.names_drop(texts)
        if 'B_stopwords' in parts:
            texts = B.stopwords_drop(texts)
        if 'B_finbert_presentation' in parts:
            texts = B.transform_to_finbert_format(texts, column_to_transform='presentation')
        if 'B_finbert_qa' in parts:
            texts = B.transform_to_finbert_format(texts, column_to_transform='q_and_a')


        # Saving
        texts.to_pickle(cf.B_C_cleaned_data)
    if 'C' in parts:
        # This part takes roughly 16 hours
        perplexities = C.LDA_perplexities([i for i in range(1, 21)])
        # C.plot_perplexities()
        # Load
        texts = pd.read_pickle(cf.B_C_cleaned_data)

        # Save
        # texts.to_pickle(cf.B_C_cleaned_data)
    if 'D' in parts:
        pass
    if 'E' in parts:
        pass
    if 'F' in parts:
        pass # See own package inside this project

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    execute_pipeline(['A',
                      'B', 'B_names', 'B_stopwords']) # General pipeline

    # For BERT
    # execute_pipeline(['B', 'B_finbert']) # For Finbert
    #execute_pipeline(['C'])
