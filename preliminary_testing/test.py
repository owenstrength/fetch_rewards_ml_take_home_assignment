import numpy as np
import pandas as pd
from tqdm import tqdm
import holidays
import matplotlib.pyplot as plt

# set random seed for reproducibility
np.random.seed(42)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0.2):
        self.W1 = np.random.randn(input_size, hidden_size1) / np.sqrt(input_size)
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) / np.sqrt(hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) / np.sqrt(hidden_size2)
        self.b3 = np.zeros((1, output_size))
        self.dropout_rate = dropout_rate

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.a1 = self.apply_dropout(self.a1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        self.a2 = self.apply_dropout(self.a2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.z3  # Linear activation for regression
        return self.a3

    def backward(self, X, y, output, learning_rate):
        m = X.shape[0]
        delta3 = output - y
        dW3 = np.dot(self.a2.T, delta3) / m
        db3 = np.sum(delta3, axis=0, keepdims=True) / m
        delta2 = np.dot(delta3, self.W3.T) * self.relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, delta2) / m
        db2 = np.sum(delta2, axis=0, keepdims=True) / m
        delta1 = np.dot(delta2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, delta1) / m
        db1 = np.sum(delta1, axis=0, keepdims=True) / m

        self.W3 -= learning_rate * dW3
        self.b3 -= learning_rate * db3
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate, patience=100):
        best_loss = np.inf
        patience_counter = 0
        
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            output = self.forward(X)
            loss = np.mean(np.square(output - y))  # Mean squared error
            
            self.backward(X, y, output, learning_rate)
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            # Learning rate decay every 1000 epochs
            if epoch % 1000 == 0 and epoch != 0:
                learning_rate *= 0.95

    def predict(self, X):
        return self.forward(X)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        return np.where(x > 0, 1, 0)

    def apply_dropout(self, X):
        mask = np.random.rand(*X.shape) > self.dropout_rate
        return X * mask / (1.0 - self.dropout_rate)

# Load and preprocess data
data = pd.read_csv('data.csv')
data['Date'] = pd.to_datetime(data['# Date'])
data.drop(columns='# Date', inplace=True)
data.set_index('Date', inplace=True)

# Add new features
data['is_weekend'] = data.index.dayofweek.isin([5, 6]).astype(int)
data['day_of_week'] = data.index.dayofweek
data['is_holiday'] = data.index.isin(holidays.US(years=data.index.year)).astype(int)
data['day_of_year'] = data.index.dayofyear
data['trend'] = np.arange(len(data))

# Normalize data
mean = data['Receipt_Count'].mean()
std = data['Receipt_Count'].std()
data['Receipt_Count_Normalized'] = (data['Receipt_Count'] - mean) / std

# Normalize other features
for column in ['is_weekend', 'day_of_week', 'is_holiday', 'day_of_year', 'trend']:
    data[f'{column}_Normalized'] = (data[column] - data[column].mean()) / data[column].std()

# Prepare sequences for training
def create_sequences(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length]
        target = data['Receipt_Count_Normalized'].iloc[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets).reshape(-1, 1)

seq_length = 150 
feature_columns = ['Receipt_Count_Normalized', 'is_weekend_Normalized', 'day_of_week_Normalized', 'is_holiday_Normalized', 'day_of_year_Normalized', 'trend_Normalized']
X, y = create_sequences(data[feature_columns], seq_length)

# Initialize and train the model
model = NeuralNetwork(input_size=seq_length * len(feature_columns), hidden_size1=128, hidden_size2=64, output_size=1)
model.train(X.reshape(X.shape[0], -1), y, epochs=5000, learning_rate=0.1, patience=200)

# Function to predict next year
def predict_next_year(model, last_sequence, mean, std, data):
    predictions = []
    current_seq = last_sequence.copy()
    
    next_year = pd.date_range(start='2022-01-01', end='2022-12-31')
    for i, date in enumerate(next_year):
        next_day_features = np.array([
            0,  # placeholder for Receipt_Count
            int(date.dayofweek in [5, 6]),
            date.dayofweek,
            int(date in holidays.US(years=[date.year])),
            date.dayofyear,
            len(data) + i  # continuation of the trend
        ])
        
        # Normalize the features
        for j, column in enumerate(feature_columns):
            next_day_features[j] = (next_day_features[j] - data[column.replace('_Normalized', '')].mean()) / data[column.replace('_Normalized', '')].std()
        
        # Replace the first feature (Receipt_Count) with the model's prediction
        next_day = model.predict(current_seq.reshape(1, -1))
        next_day_features[0] = next_day[0, 0]
        
        predictions.append(next_day[0, 0])
        current_seq = np.roll(current_seq, -len(feature_columns))
        current_seq[-len(feature_columns):] = next_day_features
    
    return (np.array(predictions) * std) + mean

# Make predictions for 2022
last_sequence = X[-1].flatten()
predictions_2022 = predict_next_year(model, last_sequence, mean, std, data)

# Aggregate daily predictions to monthly
monthly_predictions_2022 = pd.Series(predictions_2022, index=pd.date_range(start='2022-01-01', end='2022-12-31')).resample('M').sum()

print("Monthly Predictions for 2022:")
for month, prediction in zip(monthly_predictions_2022.index.strftime('%B'), monthly_predictions_2022):
    print(f"{month}: {prediction:.0f}")

# Calculate and print the total predicted receipts for 2022
total_2022 = np.sum(predictions_2022)
print(f"\nTotal predicted receipts for 2022: {total_2022:.0f}")

# Compare with 2021 total
total_2021 = data['Receipt_Count'].sum()
print(f"Total receipts for 2021: {total_2021:.0f}")
print(f"Predicted change: {((total_2022 - total_2021) / total_2021 * 100):.2f}%")

# Graph 2021 data and 2022 predictions
plt.figure(figsize=(15, 7))
plt.plot(data.index, data['Receipt_Count'], label='Actual Receipt Count (2021)')
plt.plot(pd.date_range(start='2022-01-01', end='2022-12-31'), predictions_2022, label='Predicted Receipt Count (2022)')
plt.xlabel('Date')
plt.ylabel('Receipt Count')
plt.title('Daily Actual (2021) vs Predicted (2022) Receipt Count')
plt.legend()
plt.savefig('daily_prediction_enhanced_nn.png')
plt.show()

# Graph monthly aggregates
plt.figure(figsize=(15, 7))
monthly_actual_2021 = data['Receipt_Count'].resample('M').sum()
plt.plot(monthly_actual_2021.index, monthly_actual_2021, label='Actual Monthly Receipt Count (2021)')
plt.plot(monthly_predictions_2022.index, monthly_predictions_2022, label='Predicted Monthly Receipt Count (2022)')
plt.xlabel('Month')
plt.ylabel('Receipt Count')
plt.title('Monthly Actual (2021) vs Predicted (2022) Receipt Count')
plt.legend()
plt.savefig('monthly_prediction_enhanced_nn.png')
plt.show()