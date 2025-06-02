import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import  GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
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


# Main function
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
        try:
            choice = int(input("Enter your choice (1-4): "))
        except ValueError:
            print("Invalid input. Please enter a number between 1 and 4.")
            continue

        if choice == 4:
            break
        elif choice in file_paths:
            try:
                df = load_and_preprocess_data(file_paths[choice])
                print(f"Data for {os.path.basename(file_paths[choice])} loaded successfully.")
            except FileNotFoundError:
                print(f"File not found: {file_paths[choice]}")
                continue
            except Exception as e:
                print(f"An error occurred while loading the data: {e}")
                continue

            # Train the LSTM model
            lstm_model, lstm_mse, lstm_mae, lstm_r2, lstm_y_test, lstm_y_pred, scaler = train_lstm_model(df['close'])
            print("LSTM Model Evaluation:")
            print(f"Mean Squared Error (LSTM): {lstm_mse}")
            print(f"Mean Absolute Error (LSTM): {lstm_mae}")
            print(f"R-squared (LSTM): {lstm_r2}")

            # Plot the results
            plot_results(df.index[-len(lstm_y_test):], lstm_y_test, lstm_y_pred, "LSTM")

            # Predict future prices
            future_days = int(input("Enter the number of days to predict in the future: "))
            future_predictions = predict_future_prices(lstm_model, df['close'].values, future_days, scaler)
            future_dates = pd.date_range(df.index[-1], periods=future_days + 1, freq='B')[1:]
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['close'], label='Actual Prices', color='blue')
            plt.plot(future_dates, future_predictions, label='Future Predictions (LSTM)', color='red', linestyle='dashed')
            plt.title('Stock Price Prediction with Future Forecast')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.xticks(rotation=45)
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
