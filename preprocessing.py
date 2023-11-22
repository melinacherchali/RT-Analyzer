import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import function as f

train_data = f.read_file('train.csv')
test_data = f.read_file('test.csv')

test_data = f.standardize(test_data)
test_data = f.clean_data(test_data)

train_data = f.standardize(train_data)
train_data = f.clean_data(train_data)
 

