import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data.csv')

# Data Preprocessing
data['# Date'] = pd.to_datetime(data['# Date'])
data['DateSeconds'] = data['# Date'].astype(int) / 10**9
data = data.dropna()

class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.weights

# Train on daily data
X_train = data[['DateSeconds']].values
y_train = data['Receipt_Count'].values

model = LinearRegression()
model.fit(X_train, y_train)

# Generate dates for 2022 (daily)
date_range_2022 = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')
X_2022 = np.array([(d.value / 10**9) for d in date_range_2022]).reshape(-1, 1)

# Make daily predictions for 2022
y_pred_2022 = model.predict(X_2022)

# Create DataFrame for 2022 predictions
predictions_2022 = pd.DataFrame({
    'Date': date_range_2022,
    'Predicted_Receipt_Count': y_pred_2022
})

# Aggregate 2021 data by month
data['Year_Month'] = data['# Date'].dt.to_period('M')
monthly_2021 = data.groupby('Year_Month')['Receipt_Count'].sum().reset_index()

# Aggregate 2022 predictions by month
predictions_2022['Year_Month'] = predictions_2022['Date'].dt.to_period('M')
monthly_2022 = predictions_2022.groupby('Year_Month')['Predicted_Receipt_Count'].sum().reset_index()

# Visualize Daily Results
plt.figure(figsize=(15, 7))
plt.plot(data['# Date'], data['Receipt_Count'], label='2021 Actual', color='blue')
plt.plot(predictions_2022['Date'], predictions_2022['Predicted_Receipt_Count'], label='2022 Predicted', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Daily Receipt Count')
plt.title('Daily Receipt Count: 2021 Actual vs 2022 Prediction')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('daily_receipt_count_comparison.png', dpi=300)
plt.close()

# Visualize Monthly Results
plt.figure(figsize=(15, 7))
plt.bar(monthly_2021['Year_Month'].astype(str), monthly_2021['Receipt_Count'], 
        alpha=0.8, label='2021 Actual', color='blue')
plt.bar(monthly_2022['Year_Month'].astype(str), monthly_2022['Predicted_Receipt_Count'], 
        alpha=0.8, label='2022 Predicted', color='red')
plt.xlabel('Month')
plt.ylabel('Monthly Receipt Count')
plt.title('Monthly Receipt Count: 2021 Actual vs 2022 Prediction')
plt.legend()
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('monthly_receipt_count_comparison.png', dpi=300)
plt.close()

print("Daily and monthly comparison graphs have been generated and saved.")

# Print the predictions for each month of 2022
print("\nPredicted Receipt Count for each month of 2022:")
for _, row in monthly_2022.iterrows():
    print(f"{row['Year_Month']}: {row['Predicted_Receipt_Count']:.0f}")

# Save the model weights
np.save('linear_regression_weights.npy', model.weights)