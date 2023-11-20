import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import function as f

train_data = f.read_file('train.csv')
test_data = f.read_file('test.csv')

test_data = f.standardize(test_data)
test = f.clean_data(test_data)

print ("means : ", test_data.mean(), "standart deviation : ", test_data.std())