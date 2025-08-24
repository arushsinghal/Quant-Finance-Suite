from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import streamlit as st

def anomaly_detection(ticker, start_date="2023-01-01", end_date="2025-04-17"):
    try:
        contamination = 0.05

        df = yf.download(ticker, start=start_date, end=end_date)

        df.dropna(inplace=True)

        df["Returns"] = df["Close"].pct_change().fillna(0) 
        scaler = MinMaxScaler() 
        return_scaled = scaler.fit_transform(df[["Returns"]])
        return_model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
        df["Return_Anomaly"] = return_model.fit_predict(return_scaled)

        return_anomalies = df[df["Return_Anomaly"] == -1]

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df.index, df["Returns"], label="Daily Returns", color="purple")
        ax.scatter(return_anomalies.index, return_anomalies["Returns"], color="red", label="Anomalies", marker="x")
        ax.set_title(f"{ticker} Return-Based Anomaly Detection")
        ax.set_xlabel("Date")
        ax.set_ylabel("Returns")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)
        fig.tight_layout()  

        return  return_anomalies[["Returns"]], fig 
    
    except Exception as e:
        st.error(f"Error during anomaly detection")
        return None