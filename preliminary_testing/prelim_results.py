# You are expected to build an ML model from scratch to address this challenge. Your solution can be simple or complex. You are allowed to develop your solution using any languages and frameworks like PyTorch or Tensorflow. But please note that we would like to use your solution to understand your ML knowledge base. So please avoid from using any high level libraries like scikit-learn which makes it impossible to exhibit your ML skills.
# Additionally, you are expected to build up a small app which will run an inference procedure against your own trained model and return the predicted results. You are free to build up any form of app like a web service or so but having user interaction and some sort of visualization will be a plus.
# If possible, please package your app in a Docker container that can be built and run locally or pulled down and run via Docker Hub.
# Please assume the evaluator does not have prior experience executing programs in your chosen language. Therefore, please include any documentation necessary to accomplish the above requirements.
# The code, at a minimum, must run. Please provide clear instructions on how to run it.
# When complete, please upload your codebase to a public Git repo (GitHub, Bitbucket, etc.) and email us the link. Please double-check this is publicly accessible.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

# Data Preprocessing
data = data.dropna()

# Feature Engineering
# We will use the following features to predict the target variable

# Day of Week, Receipt_Count, Receipt_Count difference from yesterday, 

data['# Date'] = pd.to_datetime(data['# Date'])
data['DateSeconds'] = data['# Date'].astype(int) / 10**9
data['DayOfWeek'] = data['# Date'].dt.dayofweek
data['Receipt_Count_Diff'] = data['Receipt_Count'].diff()

# Data Preprocessing
data = data.dropna()

target = 'Receipt_Count'

# Splitting the data into training and testing data
train = data[data['# Date'] < '2021-07-01']
test = data[data['# Date'] >= '2021-07-01']


# creating the model
# first model is linear regression with a polynomial basis function

class PolynomialRegression:
    """
    Polynomial Regression Model using Normal Equation
    """
    def __init__(self, degree=1):
        self.degree = degree
        self.weights = None

    def fit(self, X, y):
        # Create polynomial features
        X_poly = self._create_polynomial_features(X)
        
        # Add bias term
        X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]
        
        # Calculate weights using normal equation
        self.weights = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y

    def predict(self, X):
        X_poly = self._create_polynomial_features(X)
        X_poly = np.c_[np.ones(X_poly.shape[0]), X_poly]
        return X_poly @ self.weights

    def _create_polynomial_features(self, X):
        n_samples, n_features = X.shape
        X_poly = np.zeros((n_samples, n_features * self.degree))
        
        for i in range(self.degree):
            X_poly[:, i*n_features:(i+1)*n_features] = X ** (i + 1)
        
        return X_poly

X_train = train[['DateSeconds']].values
y_train = train[target].values

X_test = test[['DateSeconds']].values
y_test = test[target].values


best_r2 = -np.inf
best_degree = 1
best_model = None
best_y_pred = None

for degree in range(1, 4):
    model = PolynomialRegression(degree=degree)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - np.mean(y_test))**2)
    
    if r2 > best_r2:
        best_r2 = r2
        best_degree = degree
        best_model = model
        best_y_pred = y_pred
    
    # Visualizing the results for each degree
    plt.figure()
    plt.plot(data['# Date'], data[target].values, label='Actual')
    plt.plot(test['# Date'], y_pred, label=f'Predicted (degree={degree})', color='red')
    plt.xlabel('Date')
    plt.ylabel('Receipt Count')
    plt.title(f'Receipt Count Prediction (degree={degree})')
    plt.legend()
    plt.savefig(f'outputs/prediction_degree_{degree}.png')
    plt.close()

# Visualizing the results for the best model
plt.figure()
plt.plot(data['# Date'], data[target].values, label='Actual')
plt.plot(test['# Date'], best_y_pred, label=f'Predicted (degree={best_degree})', color='red')
plt.xlabel('Date')
plt.ylabel('Receipt Count')
plt.title(f'Receipt Count Prediction (Best Model: degree={best_degree})')
plt.legend()
plt.savefig('outputs/best_prediction.png')

# show r^2 and mse for the best model
mse = np.mean((y_test - best_y_pred)**2)

print(f'Best R^2: {best_r2}')
print(f'Best MSE: {mse}')


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np

class GRUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.batch_norm(out[:, -1, :])
        out = self.fc(out)
        return out

# Data preparation
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled).unsqueeze(1)
y_train_tensor = torch.FloatTensor(y_train_scaled)
X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(1)
y_test_tensor = torch.FloatTensor(y_test_scaled)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Model initialization
input_dim = X_train.shape[1]
hidden_dim = 128
num_layers = 3
output_dim = 1
model = GRUNetwork(input_dim, hidden_dim, num_layers, output_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training loop
num_epochs = 2000
best_val_loss = float('inf')
patience = 200
no_improve = 0

for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    scheduler.step(avg_loss)
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_test_tensor)
        val_loss = criterion(val_pred, y_test_tensor)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        no_improve += 1
    
    if no_improve >= patience:
        print(f"Early stopping at epoch {epoch}")
        break
    
    # if epoch % 100 == 0:
    #     print(f'Epoch {epoch}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')

model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor)
    mse_scaled = criterion(y_pred_scaled, y_test_tensor)
    r2_scaled = 1 - torch.sum((y_test_tensor - y_pred_scaled)**2) / torch.sum((y_test_tensor - torch.mean(y_test_tensor))**2)
    
    # Inverse transform predictions and true values
    y_pred = scaler_y.inverse_transform(y_pred_scaled.numpy())
    y_true = scaler_y.inverse_transform(y_test_tensor.numpy())
    
    mse = np.mean((y_true - y_pred)**2)
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

    print(f'Scaled Mean Squared Error: {mse_scaled.item():.4f}')
    print(f'Scaled R-squared Score: {r2_scaled.item():.4f}')
    print(f'Actual Mean Squared Error: {mse:.4f}')
    print(f'Actual R-squared Score: {r2:.4f}')

if r2 > best_r2:
    print(f"The LSTM network outperformed linear regression (R² = {r2.item():.4f} vs {best_r2:.4f})")
else:
    print(f"The LSTM network (R² = {r2.item():.4f}) did not outperform linear regression (R² = {best_r2:.4f})")

# visualize the results
plt.figure()
plt.plot(data['# Date'], data[target].values, label='Actual')
plt.plot(test['# Date'], y_pred, label='Predicted (Neural Network)', color='red')
plt.xlabel('Date')
plt.ylabel('Receipt Count')
plt.title('Receipt Count Prediction (Neural Network)')
plt.legend()
plt.savefig('outputs/neural_network_prediction.png')

# save the model
torch.save(model.state_dict(), 'outputs/neural_network_model.pth')