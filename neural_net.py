import function as f 
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
from skorch import NeuralNetRegressor
from sklearn.model_selection import RandomizedSearchCV
import skorch 

submit = False
path = 'NN_sub.csv'
plot = True

# Load data
train_data, test_data = f.read_file('train.csv','test.csv')

# Process and clean data
train_data, test_data = f.process_data(train_data,test_data)

# Separate train and test data
train_subset = int(len(train_data) * 0.8)
X, y = train_data.drop('RT', axis=1), train_data['RT']
X_train, X_test, y_train, y_test = f.split_data(train_data)

if not submit :
    X = X_train
    y = y_train

X_tensor = torch.tensor(X.values, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32).reshape(-1, 1)

# Define the neural network model using PyTorch
class NN_model(nn.Module):
    def __init__(self, input_size=X.shape[1], n_neurons=180, dropout_rate=0.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, n_neurons),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(n_neurons, 1)
        )

    def forward(self, x):
        return self.layers(x)

# create model with skorch
model_skorch = NeuralNetRegressor(
    NN_model,
    criterion=nn.MSELoss,
    optimizer=optim.Adam,
    max_epochs=500,
    batch_size=32,
    lr=0.001, 
    module__dropout_rate=0.1,  
    optimizer__weight_decay=1e-3,  
    callbacks=[('early_stopping', skorch.callbacks.EarlyStopping(patience=25))]  
)

# Make predictions using the best model
model_skorch.fit(X_tensor, y_tensor)

if submit:
    y_pred = pd.DataFrame(model_skorch.predict(torch.tensor(test_data.values,dtype=torch.float32))) # predict using test data
    f.make_submission(y_pred, path)
else:
    y_pred = pd.DataFrame(model_skorch.predict(torch.tensor(X_test.values,dtype=torch.float32))) # predict using test subset
    f.test_error(y_test,y_pred,plot) # test the error on the test subset
