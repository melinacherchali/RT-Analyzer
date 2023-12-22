import function as f 
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold,train_test_split,  cross_val_score
from skorch import NeuralNetRegressor
import skorch

# Model testing parameters 
submit = False
file_path = 'NN_sub.csv'
plot = True
CDDD = False

torch.manual_seed(42)

# Load data
train_data, test_data = f.read_file('train.csv','test.csv')

# Process and clean data
train_data, test_data = f.process_data(train_data,test_data, CDDD)

# Separate train and test data
X, y = train_data.drop('RT', axis=1), train_data['RT']
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

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
    optimizer= optim.Adam,
    max_epochs=256,
    batch_size=3,
    lr=1e-4,  
    optimizer__weight_decay=1e-3,  
    callbacks=[('early_stopping', skorch.callbacks.EarlyStopping(patience=45))]  
)

if not submit :
    #f.train_and_evaluate_model(model_skorch, X_tensor, y_tensor, X_test_tensor, y_test, plot=False)
    model_skorch.fit(X_tensor, y_tensor)
    y_pred = pd.DataFrame(model_skorch.predict(X_test_tensor))
    f.test_error(y_test, y_pred)

else :
    model_skorch.fit(X_tensor, y_tensor)
    y_pred = pd.DataFrame(model_skorch.predict(torch.tensor(test_data.values,dtype=torch.float32)))
    f.make_submission(y_pred, file_path)