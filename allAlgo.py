import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt
import os

# Define the LSTM model
def create_lstm_model(input_shape): 
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        LSTM(50), 
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Load and preprocess the data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
    df = df.sort_values('Date')
    numeric_cols = ['OPEN', 'HIGH', 'LOW', 'PREV. CLOSE', 'ltp', 'close', 'vwap', '52W H', '52W L', 'VOLUME', 'VALUE', 'No of trades']
    for col in numeric_cols:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace(',', '').astype(float)
    df = df[['Date', 'close']]
    df.set_index('Date', inplace=True)
    return df

# Train non-LSTM models
def train_non_lstm_models(X_train, X_test, y_train, y_test):
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    predictions = {} 
    metrics = {} 
    for name, model in models.items():
        model.fit(X_train, y_train) 
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics[name] = {"MSE": mse, "MAE": mae, "R2": r2}
        print(f"{name} Model Evaluation:")
        print(f"  Mean Squared Error: {mse}")
        print(f"  Mean Absolute Error: {mae}")
        print(f"  R-squared: {r2}")
    return models, predictions, metrics

# Train the LSTM model
def train_lstm_model(data, train_ratio=0.8):
    sequence_length = 60
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i, 0])
        y.append(data_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    train_size = int(len(X) * train_ratio)
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    mse = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)
    return model, mse, mae, r2, y_test_rescaled, y_pred_rescaled, scaler

# Predict future prices
def predict_future_prices(model, data, future_days, scaler, sequence_length=60):
    last_sequence = scaler.transform(data[-sequence_length:].reshape(-1, 1))
    future_predictions = []
    for _ in range(future_days):
        last_sequence_scaled = last_sequence.reshape((1, sequence_length, 1))
        next_pred = model.predict(last_sequence_scaled)[0, 0]
        future_predictions.append(next_pred)
        last_sequence = np.append(last_sequence[1:], [[next_pred]], axis=0)
    future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions_rescaled

# Plot results
def plot_results(dates, actual, predicted, model_name):
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual Prices', color='blue')
    plt.plot(dates, predicted, label=f'{model_name} Predicted Prices', color='orange', linestyle='dashed')
    plt.title(f'Stock Price Prediction ({model_name})')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    file_paths = {
        1: r'Quote-Equity-ADANIPORTS-EQ-06-09-2023-to-06-09-2024.csv',
        2: r'Quote-Equity-TATAGLOBAL-EQ-06-09-2023-to-06-09-2024.csv',
        3: r'Quote-Equity-ZOMATO-EQ-06-09-2023-to-06-09-2024.csv'
    }

    while True:
        print("Select a stock for prediction:")
        for key, value in file_paths.items():
            print(f"{key}. {os.path.basename(value)}")
        print("4. Exit")
        choice = int(input("Enter your choice (1-4): "))

        if choice == 4:
            break
        elif choice in file_paths:
            df = load_and_preprocess_data(file_paths[choice])
            print(f"Data for {os.path.basename(file_paths[choice])} loaded successfully.")

            # Split data into train and test
            train_size = int(len(df) * 0.8)
            train_data = df['close'][:train_size]
            test_data = df['close'][train_size:]

            X_train_non_lstm = np.arange(len(train_data)).reshape(-1, 1)
            y_train_non_lstm = train_data.values
            X_test_non_lstm = np.arange(len(train_data), len(df)).reshape(-1, 1)

            # Train Random Forest, Decision Tree, Gradient Boosting models
            non_lstm_predictions = {}
            non_lstm_models = {
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
            }

            for model_name, model in non_lstm_models.items():
                model.fit(X_train_non_lstm, y_train_non_lstm)
                y_pred = model.predict(X_test_non_lstm)
                non_lstm_predictions[model_name] = y_pred
                print(f"{model_name} Model Evaluation:")
                mse = mean_squared_error(test_data, y_pred)
                mae = mean_absolute_error(test_data, y_pred)
                r2 = r2_score(test_data, y_pred)
                print(f"  Mean Squared Error: {mse}")
                print(f"  Mean Absolute Error: {mae}")
                print(f"  R-squared: {r2}")

            # Train LSTM model
            lstm_model, lstm_mse, lstm_mae, lstm_r2, lstm_y_test, lstm_y_pred, scaler = train_lstm_model(df['close'])
            print("LSTM Model Evaluation:")
            print(f"Mean Squared Error (LSTM): {lstm_mse}")
            print(f"Mean Absolute Error (LSTM): {lstm_mae}")
            print(f"R-squared (LSTM): {lstm_r2}")

            # Combine predictions
            # Ensure alignment of LSTM predictions with non-LSTM predictions
            combined_predictions = np.mean(
                [non_lstm_predictions["Random Forest"], non_lstm_predictions["Decision Tree"], 
                 non_lstm_predictions["Gradient Boosting"], lstm_y_pred.reshape(-1)], axis=0)

            # Plot results
            plot_results(df.index[train_size:], test_data.values, combined_predictions, "Combined Model")

            # Predict future prices
            future_days = int(input("Enter the number of days to predict future prices: "))
            future_predictions = predict_future_prices(lstm_model, df['close'].values, future_days, scaler)
            print(f"Predicted prices for the next {future_days} days: {future_predictions.flatten()}")

if __name__ == '__main__':
    main()
