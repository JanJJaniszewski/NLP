# Data Locations
from os.path import join

Z_input = './Data/input/ReleasedDataset.zip'
Z_A_throughput = './Data/throughput/Z_A'
A_A_throughput = './Data/throughput/A_A'
A_B_throughput = './Data/throughput/A_B'
Z_A_zipfolder = './Data/throughput/Z_A/ReleasedDataset_mp3'


# File locations
texts_and_prices_file = join(A_A_throughput, 'df_texts_and_prices.pickle')
path_train = join(A_B_throughput, 'train.csv')
path_test = join(A_B_throughput, 'test.csv')
path_validate = join(A_B_throughput, 'validation.csv')

# Image location
path_images = './Data/output/images'
