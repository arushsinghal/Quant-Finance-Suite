import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import rankdata
import yfinance as yf
import pandas as pd
import numpy as np

# --- Stock Universe ---
STOCK_UNIVERSE = [
    "AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "NVDA", "JPM", "V", "UNH", "META",
    "WMT", "BAC", "PG", "XOM", "HD", "CVX", "MA", "JNJ", "LLY", "MRK",
    "CRM", "ABBV", "AVGO", "ORCL", "COST", "NEE", "TMO", "KO", "PEP", "ACN",
    "CMCSA", "DIS", "ADBE", "NFLX", "INTC", "VZ", "CSCO", "PFE", "WFC", "ABT",
    "AMD", "MCD", "LIN", "MSI", "IBM", "AMT", "DHR", "TXN", "UPS", "HON"
]


# --- Sector Data ---
STOCK_SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology", "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary", "NVDA": "Technology", "JPM": "Financials", "V": "Financials",
    "UNH": "Healthcare", "META": "Technology", "WMT": "Consumer Staples", "BAC": "Financials",
    "PG": "Consumer Staples", "XOM": "Energy", "HD": "Consumer Discretionary", "CVX": "Energy",
    "MA": "Financials", "JNJ": "Healthcare", "LLY": "Healthcare", "MRK": "Healthcare",
    "CRM": "Technology", "ABBV": "Healthcare", "AVGO": "Technology", "ORCL": "Technology", "COST": "Consumer Staples",
    "NEE": "Utilities", "TMO": "Healthcare", "KO": "Consumer Staples", "PEP": "Consumer Staples", "ACN": "Technology",
    "CMCSA": "Communication Services", "DIS": "Communication Services", "ADBE": "Technology", "NFLX": "Communication Services", "INTC": "Technology", "VZ": "Communication Services", "CSCO": "Technology", "PFE": "Healthcare", "WFC": "Financials", "ABT": "Healthcare",
    "AMD": "Technology", "MCD": "Consumer Discretionary", "LIN": "Materials", "MSI": "Technology", "IBM": "Technology", "AMT": "Real Estate", "DHR": "Healthcare", "TXN": "Technology", "UPS": "Industrials", "HON": "Industrials"
}

def get_stock_data(ticker, period="1y"):
    try:
        data = yf.download(ticker, period=period)
        if data.empty:
            st.warning(f"No data found for ticker: {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None


def calculate_indicators(df):
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    def calculate_rsi(df, period=14):
        delta = df['Close'].diff()
        up, down = delta.clip(lower=0), -delta.clip(upper=0)
        ma_up = up.rolling(window=period).mean()
        ma_down = down.rolling(window=period).mean()

        rs = np.where((ma_down != 0).any(), ma_up / ma_down, 0)

        rsi = 100 - (100 / (1 + rs))
        return rsi

    df['RSI'] = calculate_rsi(df)
    df['Momentum'] = df['Close'].diff(periods=10)

    def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
        ema_fast = df['Close'].ewm(span=fast_period, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    df['MACD'], df['Signal'], df['Histogram'] = calculate_macd(df)

    def calculate_bollinger_bands(df, period=20, num_std=2):
        middle_band = df['Close'].rolling(window=period).mean()
        upper_band = middle_band + df['Close'].rolling(window=period).std() * num_std
        lower_band = middle_band - df['Close'].rolling(window=period).std() * num_std
        return upper_band, lower_band, middle_band 

    upper_band, lower_band, middle_band = calculate_bollinger_bands(df)
    df['Upper_Band'] = upper_band
    df['Lower_Band'] = lower_band
    df['Middle_Band'] = middle_band 


    df.dropna(inplace=True)

    if len(df) < 50:  
        print("Insufficient data after removing NaN values.")  
        return None 

    return df


def prepare_data(tickers):
    stock_data = {}
    for ticker in tickers:
        data = get_stock_data(ticker)
        if data is not None:
            stock_data[ticker] = data

    processed_stocks = {}
    for ticker, df in stock_data.items():
        processed_stocks[ticker] = calculate_indicators(df)

    feature_vectors = {}
    feature_cols = ['SMA_20', 'SMA_50', 'RSI', 'Momentum', 'MACD', 'Signal', 'Histogram', 'Upper_Band', 'Lower_Band', 'Middle_Band']

    if not processed_stocks:
        st.warning("No valid stock data found.  Please check the ticker symbols.")
        return {}

    all_features = pd.DataFrame()
    for ticker, df in processed_stocks.items():
        if len(df) < 1:
            st.warning(f"Not enough data for {ticker} to calculate indicators.")
            continue 
        try:
            all_features[ticker] = df[feature_cols].iloc[-1].values
        except KeyError as e:
            st.error(f"Feature(s) missing for {ticker}: {e}.  Check if indicators are calculated correctly.")
            return {}
        except IndexError as e:
            st.error(f"Not enough data for {ticker} to get last row. Error: {e}. Increase the data period or choose a stock with more data.")
            return {}

    if all_features.empty:
        st.warning("No valid feature data could be extracted.")
        return {}

    all_features = all_features.T
    scaler = MinMaxScaler()

    scaled_features = scaler.fit_transform(all_features)

    for i, ticker in enumerate(all_features.index):
        feature_vectors[ticker] = scaled_features[i]

    return feature_vectors


def knn_recommend(target_ticker, feature_vectors, tickers_to_consider, k=5):
    if not feature_vectors:
        st.warning("No feature vectors available. Cannot perform KNN recommendation.")
        return []

    if target_ticker not in feature_vectors:
        st.warning(f"Target ticker '{target_ticker}' not found in feature vectors.")
        return []

    valid_tickers = [ticker for ticker in tickers_to_consider if ticker in feature_vectors]  

    if target_ticker not in valid_tickers:
        st.warning(f"Target ticker '{target_ticker}' is not in the valid tickers list")
        return []
    
    tickers = valid_tickers
    features = np.array([feature_vectors[ticker] for ticker in tickers])

    target_index = tickers.index(target_ticker)

    distances = euclidean_distances(features, features[target_index].reshape(1, -1)).flatten()

    distances[target_index] = np.inf  # Exclude the target ticker itself

    ranks = rankdata(distances, method='ordinal')

    ticker_distances = list(zip(tickers, distances))

    sorted_ticker_distances = sorted(ticker_distances, key=lambda x: x[1])

    nearest_neighbors = [(ticker, ranks[tickers.index(ticker)]) for ticker, distance in sorted_ticker_distances[:k]]

    return nearest_neighbors