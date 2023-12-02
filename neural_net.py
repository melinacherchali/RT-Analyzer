import function as f 
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np
from skorch import NeuralNetRegressor #sklearn + pytorch
from sklearn.model_selection import RandomizedSearchCV
import skorch 



# Load data
train_data, test_data = f.read_file('train.csv','test.csv')

# Drop categorical values
test_data = test_data.drop(['SMILES', 'mol','Compound'], axis=1)
train_data = train_data.drop(['SMILES','mol','Compound'], axis=1)

# Columns to encode
columns_to_encode = ['Lab']

# OneHotEncoder on train_data and test_data
train_data_encoded, test_data_encoded = f.apply_one_hot_encoding(train_data, test_data, columns_to_encode)

# Clean both train and test data using these columns
train_data, test_data = f.clean_data(train_data_encoded, test_data_encoded)
train_subset = int(len(train_data) * 0.8)

# Separate train and test data
X, y = train_data.drop('RT', axis=1), train_data['RT']
# X_train, X_test, y_train, y_test = f.split_data(train_data)

# standardizer = StandardScaler()
# X_standardized = standardizer.fit_transform(X)
X_tensor = torch.tensor(X.iloc[:train_subset, :].values, dtype=torch.float32)
y_tensor = torch.tensor(y.iloc[:train_subset].values, dtype=torch.float32).reshape(-1, 1)


# Define the neural network model using PyTorch
class NN_model(nn.Module):
    def __init__(self, input_size=1047, n_neurons=150, dropout_rate=0.2):
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
    max_epochs=1000,
    batch_size=32,
    lr=0.001,  # Experiment with different learning rates
    module__dropout_rate=0.3,  # Experiment with different dropout rates
    optimizer__weight_decay=1e-3,  # L2 regularization
    callbacks=[('early_stopping', skorch.callbacks.EarlyStopping(patience=25))]  # Early stopping
)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'module__n_neurons': [180, 190, 200, 210, 220],  # Adjusted around the best value
    'module__dropout_rate': [0.1, 0.2, 0.3]  # Adjusted around the best value
}

# Perform GridSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=model_skorch, param_distributions=param_grid, n_iter=10, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
random_result = random_search.fit(X_tensor, y_tensor)

print("Best MSE: %f using %s" % (random_result.best_score_, random_result.best_params_))


# Get the best model from the grid search
mach2 = random_result.best_estimator_

# Fit the best model to the data
mach2.fit(X_tensor, y_tensor)

# Make predictions
predictions = mach2.predict(torch.tensor(X.iloc[train_subset:, :].values, dtype=torch.float32))
print(np.sqrt(mean_squared_error(y.iloc[train_subset:], pd.DataFrame(predictions))))

print("Best Parameters:", random_search.best_params_)

# mach2.get_params()