import Pipeline.A_DataExtraction as A
import Pipeline.B_DataRepresentation as B
import Pipeline.C_LDA as C
import Pipeline.D_TextStructure as D
import Pipeline.E_AttentionModel as E
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
        # Loading
        texts = pd.read_pickle(cf.A_B_texts_and_prices_file)

        if 'B_pricechangesummary' in parts:
            B.create_wordcloud()
            B.price_change_summary()
            B.price_change_summary_2017() # takes quite a while

        # Executing
        if 'B_sentences' in parts:
            texts = B.sentenceindicators_drop(texts)
        if 'B_names' in parts:
            texts = B.names_drop(texts)
        if 'B_stopwords' in parts:
            texts = B.stopwords_drop(texts, keep_linebreaks=True)
        if 'B_numbers' in parts:
            texts = B.change_numbers(texts)
        if 'B_shorten' in parts:
            texts = B.shorten_texts(texts)
        if 'B_finbert' in parts:
            B.transform_to_finbert_format(texts, 'presentation', cf.path_train_pres, cf.path_test_pres, cf.path_validate_pres)
            B.transform_to_finbert_format(texts, 'q_and_a', cf.path_train_qa, cf.path_test_qa, cf.path_validate_qa)


        # Saving
        texts.to_pickle(cf.B_C_cleaned_data)
    if 'C' in parts:
        # Load
        texts = pd.read_pickle(cf.B_C_cleaned_data)

        if 'C_LDA' in parts:
            # This part takes roughly 16 hours
            perplexities = C.LDA_perplexities(texts, [i for i in range(1, 11)])
            C.plot_perplexities(perplexities)
            C.LDA(texts, num_topics=2)
            C.LDA(texts, num_topics=3)
            C.LDA(texts, num_topics=4)
            C.LDA(texts, num_topics=5)
            C.LDA(texts, num_topics=10)

        # Save
        # texts.to_pickle(cf.B_C_cleaned_data)
    if 'D' in parts:
        texts = pd.read_pickle(cf.B_C_cleaned_data)
        swem_results = D.SWEM_analysis(texts)
    if 'E' in parts:
        pass
    if 'F' in parts:
        pass  # See own package inside this project

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    execute_pipeline(['B', 'B_names', 'B_stopwords', 'B_numbers', 'B_shorten', 'B_sentences', 'B_finbert'])  # General pipeline

    # For BERT
    # execute_pipeline(['B', 'B_finbert']) # For Finbert
    #execute_pipeline(['C'])
