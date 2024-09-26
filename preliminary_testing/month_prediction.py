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
month_data = data.copy()
month_data['Month'] = month_data['# Date'].dt.month
month_data = month_data.drop(columns=['# Date'])
month_data = month_data.groupby('Month').sum()

data['DateSeconds'] = data['# Date'].astype(int) / 10**9
data['DayOfWeek'] = data['# Date'].dt.dayofweek
data['Receipt_Count_Diff'] = data['Receipt_Count'].diff()

# Data Preprocessing
data = data.dropna()

target = 'Receipt_Count'

# Splitting the data into training and testing data
train = data[data['# Date'] < '2021-12-31']
test = data[data['# Date'] >= '2021-12-31']


# creating a linear regression model using normal equation
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
    
# predict each receipt count for the next month
# predict each day and sum them for each month

model = PolynomialRegression(degree=1)

X_train = train[['DateSeconds', 'DayOfWeek']].values
y_train = train[target].values

X_test = test[['DateSeconds', 'DayOfWeek']].values
y_test = test[target].values

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

pred_data = pd.DataFrame(X_test, columns=['DateSeconds', 'DayOfWeek'])
pred_data['Predicted_Receipt_Count'] = y_pred
pred_data['# Date'] = pred_data['DateSeconds'].apply(lambda x: pd.Timestamp(x, unit='s'))
pred_data = pred_data.drop(columns=['DateSeconds'])

# calculate the error
error = np.mean((y_pred - y_test) ** 2)
r2 = 1 - (error / np.var(y_test))
print(f'Mean Squared Error: {error}')
print(f'R2 Score: {r2}')

# predicted monthly receipt count
pred_data['Month'] = pred_data['# Date'].dt.month
pred_data = pred_data.drop(columns=['# Date'])
pred_data = pred_data.groupby('Month').sum()
print(pred_data)
truth_month_data = month_data.copy()
print(truth_month_data)

# graph pred_data vs truth_month_data

plt.plot(pred_data.index, pred_data['Predicted_Receipt_Count'], label='Predicted Receipt Count')
plt.plot(truth_month_data.index, truth_month_data['Receipt_Count'], label='Actual Receipt Count')
plt.xlabel('Month')
plt.ylabel('Receipt Count')
plt.title('Predicted vs Actual Monthly Receipt Count using Linear Regression')
plt.legend()
# save 
plt.savefig('month_prediction.png')

# 2022-01-01 to 2022-12-31 prediction

# predict each receipt count for the next month
# predict each day and sum them for each month

# create a new dataframe with the dates from 2022-01-01 to 2022-12-31

date_range = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
pred22_data = pd.DataFrame(date_range, columns=['# Date'])
pred22_data['DateSeconds'] = pred22_data['# Date'].astype(int) / 10**9
pred22_data['DayOfWeek'] = pred22_data['# Date'].dt.dayofweek
pred22_data['month'] = pred22_data['# Date'].dt.month


X_pred22 = pred22_data[['DateSeconds', 'DayOfWeek']].values
y_pred22 = model.predict(X_pred22)

pred22_data = pred22_data.drop(columns=['# Date', 'DayOfWeek'])
pred22_data['Predicted_Receipt_Count'] = y_pred22
pred22_data = pred22_data.groupby('month').sum()

# for every index in pred22_data, add 12 to it
pred22_data.index = pred22_data.index + 12
print(pred22_data['Predicted_Receipt_Count'])

# graph pred22_data

plt.plot(pred22_data.index, pred22_data['Predicted_Receipt_Count'], label='Predicted 2022 Receipt Count', color='red')
plt.xlabel('Month')
plt.ylabel('Receipt Count')
plt.title('Predicted Monthly Receipt Count for 2022 using Linear Regression')
plt.legend()
# save
plt.savefig('2022_month_prediction.png')


# same thing but using this gru model
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

# predicted monthly receipt count
# Predict for the year 2022
date_range = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
pred22_data = pd.DataFrame(date_range, columns=['# Date'])
pred22_data['DateSeconds'] = pred22_data['# Date'].astype(int) / 10**9
pred22_data['DayOfWeek'] = pred22_data['# Date'].dt.dayofweek
pred22_data['month'] = pred22_data['# Date'].dt.month

X_pred22 = pred22_data[['DateSeconds', 'DayOfWeek']].values
X_pred22_scaled = scaler_X.transform(X_pred22)
X_pred22_tensor = torch.FloatTensor(X_pred22_scaled).unsqueeze(1)

with torch.no_grad():
    y_pred22_scaled = model(X_pred22_tensor)
    y_pred22 = scaler_y.inverse_transform(y_pred22_scaled.numpy())

pred22_data = pred22_data.drop(columns=['# Date', 'DayOfWeek'])
pred22_data['Predicted_Receipt_Count'] = y_pred22
pred22_data = pred22_data.groupby('month').sum()

# Adjust the index to represent the months in 2022
pred22_data.index = pred22_data.index + 12
print(pred22_data['Predicted_Receipt_Count'])

# graph pred_data vs truth_month_data

plt.close()

plt.plot(pred22_data.index, pred22_data['Predicted_Receipt_Count'], label='Predicted Receipt Count')
plt.plot(truth_month_data.index, truth_month_data['Receipt_Count'], label='Actual Receipt Count')
plt.xlabel('Month')
plt.ylabel('Receipt Count')
plt.title('Predicted vs Actual Monthly Receipt Count using GRU')
plt.legend()
# save
plt.savefig('gru_month_prediction.png')