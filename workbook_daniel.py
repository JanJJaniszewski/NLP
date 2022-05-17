import config as cf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#%%

texts = pd.read_pickle(cf.texts_and_prices_file)

call_i = texts.call[0]


