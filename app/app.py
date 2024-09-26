import os
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler # this just standardizes the data, it doesn't any ml computations
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import plotly
import plotly.graph_objs as go
import json

# Linear Regression Model Using Normal Equation
class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.weights

# GRU Model
class GRUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GRUNetwork, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

# Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Helper Function for Allowed File Types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Engineered Features
def process_data(file_path):
    data = pd.read_csv(file_path)
    data['# Date'] = pd.to_datetime(data['# Date'])
    data['DateSeconds'] = data['# Date'].astype(int) / 10**9
    data['DayOfWeek'] = data['# Date'].dt.dayofweek
    data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
    holidays = pd.to_datetime(['2021-01-01', '2021-12-25', '2021-07-04'])
    data['IsHoliday'] = data['# Date'].isin(holidays).astype(int)
    return data

def train_models(data):
    # Linear Regression
    X_train_lr = data[['DateSeconds', 'DayOfWeek', 'IsWeekend', 'IsHoliday']].values
    y_train_lr = data['Receipt_Count'].values
    lr_model = LinearRegression()
    lr_model.fit(X_train_lr, y_train_lr)


    # GRU
    # Standardize data
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X = scaler_X.fit_transform(data[['DateSeconds', 'DayOfWeek', 'IsWeekend', 'IsHoliday']].values)
    y = scaler_y.fit_transform(data[['Receipt_Count']].values)

    X_tensor_GRU = torch.FloatTensor(X).unsqueeze(1)
    y_tensor_GRU = torch.FloatTensor(y)
    
    gru_model = GRUNetwork(input_dim=4, hidden_dim=64, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(gru_model.parameters(), lr=0.001)

    batch_size = 64
    num_epochs = 1000

    # Prepare DataLoader
    dataset = TensorDataset(X_tensor_GRU, y_tensor_GRU)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        gru_model.train()
        for batch_X, batch_y in data_loader:
            optimizer.zero_grad()
            outputs = gru_model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return lr_model, gru_model, scaler_X, scaler_y

def predict_2022(lr_model, gru_model, scaler_X, scaler_y, data_2021):
    # create 2022 data
    date_range_2022 = pd.date_range(start='2022-01-01', end='2022-12-31', freq='D')

    X_2022 = pd.DataFrame({
        'DateSeconds': date_range_2022.astype(int) / 10**9,
        'DayOfWeek': date_range_2022.dayofweek,
        'IsWeekend': date_range_2022.dayofweek.isin([5, 6]).astype(int),
        'IsHoliday': date_range_2022.isin(pd.to_datetime(['2022-01-01', '2022-12-25', '2022-07-04'])).astype(int)
    })

    # predict 2022 data
    lr_predictions = lr_model.predict(X_2022.values)

    X_2022_scaled = scaler_X.transform(X_2022)
    X_2022_gru_tensor = torch.FloatTensor(X_2022_scaled).unsqueeze(1)
    
    gru_model.eval()
    with torch.no_grad():
        gru_predictions = gru_model(X_2022_gru_tensor).numpy()
        gru_predictions = scaler_y.inverse_transform(gru_predictions).flatten()

    # Create DataFrame for 2022 predictions
    predictions = pd.DataFrame({
        'Date': date_range_2022,
        'LR_Prediction': lr_predictions.flatten(),
        'GRU_Prediction': gru_predictions
    })

    # Combine 2021 data with 2022 predictions
    combined_data = pd.concat([
        data_2021[['# Date', 'Receipt_Count']].rename(columns={'# Date': 'Date'}),
        predictions
    ]).sort_values('Date')

    # Create monthly aggregations
    monthly_data = combined_data.groupby(combined_data['Date'].dt.to_period('M')).agg({
        'Receipt_Count': 'sum',
        'LR_Prediction': 'sum',
        'GRU_Prediction': 'sum'
    }).reset_index()
    monthly_data['Date'] = monthly_data['Date'].astype(str)

    return combined_data, monthly_data

def create_graphs(combined_data, monthly_data):
    # Daily Linear Regression Graph
    daily_lr = go.Figure()
    daily_lr.add_trace(go.Scatter(x=combined_data['Date'], y=combined_data['Receipt_Count'], name='2021 Actual', mode='lines'))
    daily_lr.add_trace(go.Scatter(x=combined_data['Date'], y=combined_data['LR_Prediction'], name='LR Prediction', mode='lines'))
    daily_lr.update_layout(title='Daily Receipt Count: Actual vs Linear Regression Prediction', xaxis_title='Date', yaxis_title='Receipt Count')

    # Monthly Linear Regression Graph
    monthly_lr = go.Figure()
    monthly_lr.add_trace(go.Bar(x=monthly_data['Date'], y=monthly_data['Receipt_Count'], name='2021 Actual'))
    monthly_lr.add_trace(go.Bar(x=monthly_data['Date'], y=monthly_data['LR_Prediction'], name='LR Prediction'))
    monthly_lr.update_layout(title='Monthly Receipt Count: Actual vs Linear Regression Prediction', xaxis_title='Month', yaxis_title='Receipt Count', barmode='group')

    # Daily GRU Graph
    daily_gru = go.Figure()
    daily_gru.add_trace(go.Scatter(x=combined_data['Date'], y=combined_data['Receipt_Count'], name='2021 Actual', mode='lines'))
    daily_gru.add_trace(go.Scatter(x=combined_data['Date'], y=combined_data['GRU_Prediction'], name='GRU Prediction', mode='lines'))
    daily_gru.update_layout(title='Daily Receipt Count: Actual vs GRU Prediction', xaxis_title='Date', yaxis_title='Receipt Count')

    # Monthly GRU Graph
    monthly_gru = go.Figure()
    monthly_gru.add_trace(go.Bar(x=monthly_data['Date'], y=monthly_data['Receipt_Count'], name='2021 Actual'))
    monthly_gru.add_trace(go.Bar(x=monthly_data['Date'], y=monthly_data['GRU_Prediction'], name='GRU Prediction'))
    monthly_gru.update_layout(title='Monthly Receipt Count: Actual vs GRU Prediction', xaxis_title='Month', yaxis_title='Receipt Count', barmode='group')

    return {
        'daily_lr': json.dumps(daily_lr, cls=plotly.utils.PlotlyJSONEncoder),
        'monthly_lr': json.dumps(monthly_lr, cls=plotly.utils.PlotlyJSONEncoder),
        'daily_gru': json.dumps(daily_gru, cls=plotly.utils.PlotlyJSONEncoder),
        'monthly_gru': json.dumps(monthly_gru, cls=plotly.utils.PlotlyJSONEncoder)
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            file_path = 'data.csv'  # Use default dataset
        file = request.files['file']
        if file.filename == '':
            file_path = 'data.csv'  # Use default dataset
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        else:
            file_path = 'data.csv'  # Use default dataset

        data = process_data(file_path)
        lr_model, gru_model, scaler_X, scaler_y = train_models(data)
        combined_data, monthly_data = predict_2022(lr_model, gru_model, scaler_X, scaler_y, data)
        graphs = create_graphs(combined_data, monthly_data)

        # Filter monthly_data for 2022 only
        monthly_data_2022 = monthly_data[monthly_data['Date'].str.startswith('2022')]
        print(monthly_data.head(20))

        return render_template('results.html', predictions=monthly_data_2022.to_dict('records'), graphs=graphs)

    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=8001)  