# Data Locations
from os.path import join

Z_input = './Data/input/ReleasedDataset.zip'
Z_A_throughput = './Data/throughput/Z_A'
A_A_throughput = './Data/throughput/A_A'
A_B_throughput = './Data/throughput/A_B'
B_C_throughput = './Data/throughput/B_C'
C_C_throughput = './Data/throughput/C_C'
Z_A_zipfolder = './Data/throughput/Z_A/ReleasedDataset_mp3'


# File locations
A_B_texts_and_prices_file = join(A_B_throughput, 'df_texts_and_prices.pickle')
B_C_cleaned_data = join(B_C_throughput, 'df_texts_and_prices.pickle')
path_nlp = './finBERT-master/data/sentiment_data'
path_train_qa = join(path_nlp, 'train_qa.csv')
path_test_qa = join(path_nlp, 'test_qa.csv')
path_validate_qa = join(path_nlp, 'validation_qa.csv')
path_train_pres = join(path_nlp, 'train_pres.csv')
path_test_pres = join(path_nlp, 'test_pres.csv')
path_validate_pres = join(path_nlp, 'validation_pres.csv')
C_lda_perplexities = join(C_C_throughput, 'perplexities.pickle')
C_lda_2_topics = join(C_C_throughput, 'lda_2_topics.pickle')
C_lda_3_topics = join(C_C_throughput, 'lda_3_topics.pickle')
C_lda_model = join(C_C_throughput, 'lda_model')

# Image location
path_images = './Data/output/images'
