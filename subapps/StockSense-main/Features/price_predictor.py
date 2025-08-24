from datetime import datetime
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

START_DATE = "2020-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')
LOOK_BACK = 10
EPOCHS = 100
BATCH_SIZE = 32
FUTURE_DAYS = 1

def fetch_stock_data(symbol):
    df = yf.download(symbol, start=START_DATE, end=END_DATE)
    if df.empty:
        raise Exception("No data found for the specified symbol and dates.")
    return df

def add_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    RS = up.rolling(14).mean() / down.rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + RS))
    EMA_12 = df['Close'].ewm(span=12).mean()
    EMA_26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = EMA_12 - EMA_26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9).mean()
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
    low14 = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['STOCH_K'] = 100 * (df['Close'] - low14) / (high14 - low14)
    df['STOCH_D'] = df['STOCH_K'].rolling(3).mean()
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df.dropna(inplace=True)
    return df

def preprocess(df):
    features = ['Close', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_SIGNAL', 'MACD_HIST', 'STOCH_K', 'STOCH_D', 'OBV']
    data = df[features].values

    if data.size == 0:
        raise ValueError("No data available after feature selection.")

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def create_dataset(data, look_back):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        LSTM(100),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_future_dataset(model, last_seq, scaler, look_back, days, num_features):
    future_predictions_scaled = []
    current_sequence = last_seq.copy()

    for _ in range(days):
        input_sequence = np.reshape(current_sequence, (1, look_back, num_features))
        predicted_scaled = model.predict(input_sequence, verbose=0)[0, 0]
        future_predictions_scaled.append(predicted_scaled)

        new_row = np.zeros(num_features)
        new_row[0] = predicted_scaled
        current_sequence = np.vstack((current_sequence[1:], new_row))

    future_predictions_scaled = np.array(future_predictions_scaled).reshape(-1, 1)
    dummy_features = np.zeros((future_predictions_scaled.shape[0], num_features - 1))
    inverted_data = np.concatenate((future_predictions_scaled, dummy_features), axis=1)
    future_prices = scaler.inverse_transform(inverted_data)[:, 0]
    return future_prices

@st.cache_resource(ttl=86400)  # Cache for 1 day
def get_or_train_model(symbol, X_train, y_train, input_shape):
    model = build_model(input_shape)
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    return model

def run_stock_prediction(symbol):
    df = fetch_stock_data(symbol)
    df = add_indicators(df)
    scaled_data, scaler = preprocess(df)
    X, y = create_dataset(scaled_data, LOOK_BACK)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = get_or_train_model(symbol, X_train, y_train, (LOOK_BACK, X.shape[2]))

    predicted_test = model.predict(X_test, verbose=0)
    predicted_test_prices = scaler.inverse_transform(np.column_stack((predicted_test, np.zeros((len(predicted_test), X.shape[2]-1)))))[:, 0]
    true_test_prices = scaler.inverse_transform(np.column_stack((y_test.reshape(-1,1), np.zeros((len(y_test), X.shape[2]-1)))))[:, 0]

    rmse = np.sqrt(mean_squared_error(true_test_prices, predicted_test_prices))
    nrmse = rmse / (np.max(true_test_prices) - np.min(true_test_prices))
    mape = mean_absolute_percentage_error(true_test_prices, predicted_test_prices)
    r2 = r2_score(true_test_prices, predicted_test_prices)

    last_sequence = scaled_data[-LOOK_BACK:]
    predicted_price = create_future_dataset(model, last_sequence, scaler, LOOK_BACK, FUTURE_DAYS, X.shape[2])[0]
    previous_day_price = df['Close'].iloc[-1].item()

    trend = "Increase 📈" if predicted_price > previous_day_price else "Decrease 📉"

    result = {
        "symbol": symbol,
        "nrmse": nrmse,
        "mape": mape,
        "r2": r2,
        "predicted_price": predicted_price,
        "previous_price": previous_day_price,
        "trend": trend
    }

    fig1 = plt.figure(figsize=(12, 6))
    plt.plot(true_test_prices, label='Actual Prices', color='blue')
    plt.plot(predicted_test_prices, label='Predicted Prices', color='orange')
    plt.title(f"{symbol} - Actual vs Predicted Stock Prices (Test Set)")
    plt.xlabel("Time") 
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig1)

    return result
