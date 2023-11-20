import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import function as f

train_data = f.read_file('train.csv')
test_data = f.read_file('test.csv')

train_data = f.standardize(train_data)
#f.clean_data([train_data, test_data])

print ("means : ", train_data.mean(), "standart deviation : ", train_data.std())