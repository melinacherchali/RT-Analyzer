import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import skorch
import random
from xgboost import XGBRegressor

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, concatenate, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetRegressor

import function as f



def keras_nn(X,
             y,
             test_data,
             submit=False,
             path='keras.csv',
             plot=True
             ):
    """
    Trains a two-branch neural network model using Keras.

    Args:
        X (Dataframe): The input data for training the model.
        y (Dataframe): The target data for training the model.
        test_data (Dataframe): The input data for testing the model.
        submit (bool, optional): Whether to submit the predictions. Defaults to False.
        path (str, optional): The file path for submitting the predictions. Defaults to 'keras.csv'.
        plot (bool, optional): Whether to plot the test error. Defaults to True.

    Returns:
        None
    """

    # Set the seed for reproducibility
    seed_value = 42
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

    # Set the random seed for reproducible results
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # Set deterministic behavior for CuDNN
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    # GPU configuration
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = '0'  # Adjust GPU index if needed
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Assuming in_dim is the number of input features
    if submit:
        in_dim = X.shape[1]
    else:
        in_dim = X_train.shape[1]

    # Define the input layers
    input1 = Input(shape=(in_dim,))
    input2 = Input(shape=(in_dim,))

    unit = 512
    unit2 = 256
    unit3 = 128
    # Define the first branch (Q1)
    Q1 = Dense(units=unit, activation='relu')(input1)
    Q1 = Dense(units=unit, activation='relu')(Q1)
    Q1 = Dense(units=unit, activation='relu')(Q1)

    # Define the second branch (Q2)
    Q2 = Dense(units=unit2, activation='relu')(input2)
    Q2 = Dense(units=unit2, activation='relu')(Q2)
    Q2 = Dense(units=unit2, activation='relu')(Q2) 

    # Define the third branch (Q3)
    Q3 = Dense(units=unit3, activation='relu')(input2)
    Q3 = Dense(units=unit3, activation='relu')(Q2)
    Q3 = Dense(units=unit3, activation='relu')(Q2) 
    
    
    # Concatenate the outputs of both branches
    pred = concatenate([Q1, Q2, Q3])
    pred = Dense(units=1, activation='linear')(pred)

    # Create the model with two inputs and one output
    model_twoBranch = Model(inputs=[input1, input2], outputs=pred)

    # Compile the model
    model_twoBranch.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))

    # Display the model summary
    model_twoBranch.summary()

    # Define the early stopping callback
    early_stop = EarlyStopping(monitor='loss', patience=15, mode='min', min_delta=0.01, restore_best_weights=False)

    # Fit the model to the training data
    if submit:
        model_twoBranch.fit(x=[X, X], y=y, validation_split=0.1, batch_size=32, epochs=100, callbacks=[early_stop])
        test_y_pred = pd.DataFrame(model_twoBranch.predict([test_data, test_data]))
        f.make_submission(test_y_pred, path)

    else:
        model_twoBranch.fit(x=[X_train, X_train], y=y_train, validation_split=0.1, batch_size=16, epochs=100, callbacks=[early_stop])
        test_y_pred = pd.DataFrame(model_twoBranch.predict([X_test, X_test]))
        f.test_error(y_test,test_y_pred,plot)
        # f.make_submission(test_y_pred, path)



def pytorch_nn(X,
              y,
              test_data,
              submit=False, 
              path='nn.csv', 
              plot=True):
    """
    Trains a PyTorch neural network model using the given data.

    Parameters:
    - X (pandas.DataFrame): The input features for training the model.
    - y (pandas.Series): The target variable for training the model.
    - test_data (pandas.DataFrame): The input features for testing the model.
    - submit (bool, optional): Whether to generate a submission file. Defaults to False.
    - path (str, optional): The file path for the submission file. Defaults to 'nn.csv'.
    - plot (bool, optional): Whether to plot the training and validation loss. Defaults to True.
    """
        
    torch.manual_seed(42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    if submit:
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)
    else:
        X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
        y_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)

    # Define the neural network model using PyTorch
    class NN_model(nn.Module):
        def __init__(self, input_size=X.shape[1], dropout_rate=0.2):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 750),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(750, 500),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(500, 250),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(250, 1)
            )

        def forward(self, x):
            return self.layers(x)

    # create model with skorch
    model_skorch = NeuralNetRegressor(
        NN_model,
        criterion=nn.MSELoss,
        optimizer=optim.Adam,
        max_epochs=256,
        batch_size=3,
        lr=1e-4,  
        optimizer__weight_decay=1e-3,  
        callbacks=[('early_stopping', skorch.callbacks.EarlyStopping(patience=45))]  
    )

    # Fit model
    model_skorch.fit(X_tensor, y_tensor)

    if submit:
        model_skorch.fit(X_tensor, y_tensor)
        y_pred = pd.DataFrame(model_skorch.predict(torch.tensor(test_data.values, dtype=torch.float32)))
        f.make_submission(y_pred, path)
    else:
        #f.train_and_evaluate_model(model_skorch, X_tensor, y_tensor, X_test_tensor, y_test, plot=False)
        model_skorch.fit(X_tensor, y_tensor)
        y_pred = pd.DataFrame(model_skorch.predict(X_test_tensor))
        f.test_error(y_test, y_pred) s
        
def gradient_boost(X,
                   y,
                   test_data,
                   submit=False,
                   path='gb.csv',
                   plot=True):
    """
    Trains a gradient boosting model and makes predictions.

    Parameters:
    - X (array-like): The input features for training the model.
    - y (array-like): The target variable for training the model.
    - test_data (array-like): The input features for making predictions.
    - submit (bool, optional): Whether to make submission predictions or not. Default is False.
    - path (str, optional): The file path for saving the submission predictions. Default is 'gb.csv'.
    - plot (bool, optional): Whether to plot the test error or not. Default is True.

    Returns:
    - None
    """

    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # Define model
    model_GB =  XGBRegressor(subsample=0.843, 
                             reg_lambda=1, 
                             reg_alpha=0, 
                             n_estimators=2500, 
                             max_depth=4, 
                             learning_rate=0.11, 
                             colsample_bytree=0.8055,
                             seed=random_seed)

    if not submit:
        model_GB.fit(X_train, y_train)
        y_pred = pd.DataFrame(model_GB.predict(X_test))
        f.test_error(y_test, y_pred, plot)
    else:
        model_GB.fit(X, y) 
        y_pred = pd.DataFrame(model_GB.predict(test_data)) 
        f.make_submission(y_pred, path)
    
    
# Load data
# train_data, test_data = f.read_file('train.csv','test.csv')

# # Process and clean data
# train_data, test_data = f.process_data(train_data,test_data)

# # Separate train and test data
# X, y = train_data.drop('RT', axis=1), train_data['RT']

# keras_nn(X, y, test_data, submit=False, path='keras_combo_test.csv', plot=False)
# pytorch_nn(X, y, test_data, submit=False, path='nn_combo_test.csv', plot=False)
# gradient_boost(X, y, test_data, submit=False, path='gb_combo_test.csv', plot=False)
