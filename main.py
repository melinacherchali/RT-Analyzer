import function as f
import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import models 


# Set path of model submissions
neuralnet = 'nn.csv'
keras_neuralnet = 'keras.csv'
gb = 'gb.csv'

# Load data
train_data, test_data = f.read_file('train.csv','test.csv')

# Process and clean data
train_data, test_data = f.process_data(train_data,test_data)

# Separate train and test data
X, y = train_data.drop('RT', axis=1), train_data['RT']

# Train models and make submissions
models.pytorch_nn(X, y, test_data, submit=True, path=neuralnet, plot=False)
models.keras_nn(X, y, test_data, submit=True, path=keras_neuralnet, plot=False)
models.gradient_boost(X, y, test_data, submit=True, path=gb, plot=False)

# Average the predictions
nn_pred = pd.read_csv(neuralnet)
keras_pred = pd.read_csv(keras_neuralnet)
gb_pred = pd.read_csv(gb)
pred = (nn_pred['RT'] + keras_pred['RT'] + gb_pred['RT']) / 3

# Convert to DataFrame
pred_df = pd.DataFrame(pred, columns=['RT'])

# Make submission
f.make_submission(pred_df, 'submission.csv')



