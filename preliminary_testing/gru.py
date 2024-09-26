import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
data = pd.read_csv('data.csv')
data['# Date'] = pd.to_datetime(data['# Date'])
data['DateSeconds'] = data['# Date'].astype(int) / 10**9
data['DayOfWeek'] = data['# Date'].dt.dayofweek
data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)

# Assuming you have a list of holidays
holidays = pd.to_datetime([
    '2021-01-01', '2021-12-25', '2021-07-04',
])
data['IsHoliday'] = data['# Date'].isin(holidays).astype(int)
data = data.dropna()

# Prepare data for GRU model
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(data[['DateSeconds', 'DayOfWeek', 'IsWeekend', 'IsHoliday']].values)
y = scaler_y.fit_transform(data[['Receipt_Count']].values)

X_tensor = torch.FloatTensor(X).unsqueeze(1)
y_tensor = torch.FloatTensor(y)

# GRU model definition
class GRUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Model parameters
input_dim = 4
hidden_dim = 64
output_dim = 1
model = GRUNetwork(input_dim, hidden_dim, output_dim)

# Training parameters
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000
batch_size = 64

# Prepare DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in data_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Generate predictions for 2022
date_range_2022 = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
X_2022 = pd.DataFrame({
    'DateSeconds': date_range_2022.astype(int) / 10**9,
    'DayOfWeek': date_range_2022.dayofweek,
    'IsWeekend': date_range_2022.dayofweek.isin([5, 6]).astype(int),
    'IsHoliday': date_range_2022.isin(holidays).astype(int)
})

X_2022_scaled = scaler_X.transform(X_2022)
X_2022_tensor = torch.FloatTensor(X_2022_scaled).unsqueeze(1)

model.eval()
with torch.no_grad():
    y_pred_2022_scaled = model(X_2022_tensor).numpy()
    y_pred_2022 = scaler_y.inverse_transform(y_pred_2022_scaled)

# Create DataFrame for 2022 predictions
predictions_2022 = pd.DataFrame({
    'Date': date_range_2022,
    'Predicted_Receipt_Count': y_pred_2022.flatten()
})

# Aggregate 2021 data and 2022 predictions by month
data['Year_Month'] = data['# Date'].dt.to_period('M')
monthly_2021 = data.groupby('Year_Month')['Receipt_Count'].sum().reset_index()

predictions_2022['Year_Month'] = predictions_2022['Date'].dt.to_period('M')
monthly_2022 = predictions_2022.groupby('Year_Month')['Predicted_Receipt_Count'].sum().reset_index()

# Visualize Daily Results
plt.figure(figsize=(15, 7))
plt.plot(data['# Date'], data['Receipt_Count'], label='2021 Actual', color='blue')
plt.plot(predictions_2022['Date'], predictions_2022['Predicted_Receipt_Count'], label='2022 Predicted', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Daily Receipt Count')
plt.title('Daily Receipt Count: 2021 Actual vs 2022 Prediction (GRU Model)')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('daily_receipt_count_comparison_gru.png', dpi=300)
plt.close()

# Visualize Monthly Results
plt.figure(figsize=(15, 7))
plt.bar(monthly_2021['Year_Month'].astype(str), monthly_2021['Receipt_Count'], 
        alpha=0.8, label='2021 Actual', color='blue')
plt.bar(monthly_2022['Year_Month'].astype(str), monthly_2022['Predicted_Receipt_Count'], 
        alpha=0.8, label='2022 Predicted', color='red')
plt.xlabel('Month')
plt.ylabel('Monthly Receipt Count')
plt.title('Monthly Receipt Count: 2021 Actual vs 2022 Prediction (GRU Model)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('monthly_receipt_count_comparison_gru.png', dpi=300)
plt.close()

print("Daily and monthly comparison graphs for GRU model have been generated and saved.")

# Print the predictions for each month of 2022
print("\nPredicted Receipt Count for each month of 2022 (GRU Model):")
for _, row in monthly_2022.iterrows():
    print(f"{row['Year_Month']}: {row['Predicted_Receipt_Count']:.0f}")

# Save the model
torch.save(model.state_dict(), 'gru_model.pth')