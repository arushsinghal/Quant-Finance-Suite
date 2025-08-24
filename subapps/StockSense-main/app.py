from datetime import datetime
import pandas as pd
import streamlit as st
from Features.anomaly_detect import anomaly_detection
from Features.price_predictor import run_stock_prediction
import Features.similar_stocks as similar_stocks
from scipy.stats import rankdata
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import numpy as np

st.set_page_config(page_title="Stock Dashboard", layout="wide")


st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["📊 Stock Prediction", "🤝 Similar Stocks",  "🚨 Anomaly Detection"])


@st.cache_resource(ttl=86400)
def get_feature_vectors():
    return similar_stocks.prepare_data(similar_stocks.STOCK_UNIVERSE)


if page == "📊 Stock Prediction":
    st.title("📊 Stock Price Predictor")

    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT, GOOG):", value="GOOG")

    if st.button("Predict"):
        with st.spinner("Analysing..."):
            try:
                result = run_stock_prediction(stock_symbol.upper())
                st.success(f"Prediction for {result['symbol'].upper()}")

                col1, col2 = st.columns(2)
                with col1:
                    currency_symbol = "₹" if stock_symbol.upper().endswith(".NS") else "$"
                    st.metric("Previous Close", f"{currency_symbol}{result['previous_price']:.2f}")
                with col2:
                    st.metric(
                        "Predicted Close",
                        f"{currency_symbol}{result['predicted_price']:.2f}",
                        delta=f"{result['predicted_price'] - result['previous_price']:.2f}"
                    )

                st.write("### Price Comparison")
                fig = go.Figure()
                trend_color = "green" if result["predicted_price"] > result["previous_price"] else "red"
                trend_text = "Increase" if result["predicted_price"] > result["previous_price"] else "Decrease"
                fig.add_trace(go.Scatter(
                    x=["Previous", "Predicted"],
                    y=[result["previous_price"], result["predicted_price"]],
                    mode="lines+markers",
                    name="Price Trend",
                    line=dict(color=trend_color) 
                ))
                fig.add_annotation(
                    x="Predicted",
                    y=result["predicted_price"],
                    text=trend_text,
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40,
                    font=dict(color=trend_color)
                )
                fig.update_layout(
                    title="Previous vs Predicted Prices",
                    xaxis_title="Type",
                    yaxis_title="Price ($)"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("🧠 Model Evaluation")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(label="MAPE", value=f"{result['mape']:.4f}")
                with col2:
                    st.metric(label="R² Score", value=f"{result['r2']:.4f}")
                with col3:
                    st.metric(label="NRMSE", value=f"{result['nrmse']:.4f}")
            except Exception as e:
                st.error(f"Error: {e}")

elif page == "🤝 Similar Stocks":
    st.title("🤝 Similar Stock Recommender")

    selected_ticker = st.selectbox("Select a Stock:", similar_stocks.STOCK_UNIVERSE)
    selected_sectors = st.multiselect(
        "Filter by Sectors (Optional):",
        sorted(list(set(similar_stocks.STOCK_SECTORS.values())))
    )

    if st.button("Get Recommendations"):
        with st.spinner("Processing..."):
            feature_vectors = get_feature_vectors()

        if feature_vectors:
            if selected_sectors:
                tickers_to_consider = [
                    ticker for ticker in similar_stocks.STOCK_UNIVERSE
                    if similar_stocks.STOCK_SECTORS.get(ticker) in selected_sectors or ticker == selected_ticker
                ]
            else:
                tickers_to_consider = similar_stocks.STOCK_UNIVERSE

            recommendations = similar_stocks.knn_recommend(selected_ticker, feature_vectors, tickers_to_consider)

            if recommendations:
                st.subheader(f"Top 5 Recommendations for {selected_ticker}:")

                recommendations.sort(key=lambda item: item[1])
                ranks = rankdata([distance for _, distance in recommendations], method='ordinal')

                for i in range(min(5, len(recommendations))):
                    ticker, dist = recommendations[i]
                    rank = int(ranks[i]) 
                    st.write(f"- {ticker} (Rank: {rank}) - Sector: {similar_stocks.STOCK_SECTORS.get(ticker, 'N/A')}")

                if len(recommendations) < 5:
                    st.info(f"Only {len(recommendations)} recommendations found.")
                    for j in range(len(recommendations), 5):
                        st.write("- N/A")

                # Visualize in 3D using PCA
                tickers = [selected_ticker] + [ticker for ticker, _ in recommendations]
                vectors = np.array([feature_vectors[ticker] for ticker in tickers])

                # Reduce to 3D
                pca = PCA(n_components=3)
                reduced_vectors = pca.fit_transform(vectors)

                fig = go.Figure()

                # Selected Stock
                fig.add_trace(go.Scatter3d(
                    x=[reduced_vectors[0][0]],
                    y=[reduced_vectors[0][1]],
                    z=[reduced_vectors[0][2]],
                    mode='markers+text',
                    marker=dict(size=10, color='red'),
                    name=selected_ticker,
                    text=[selected_ticker],
                    textposition='top center'
                ))

                for i in range(1, len(reduced_vectors)):
                    fig.add_trace(go.Scatter3d(
                        x=[reduced_vectors[i][0]],
                        y=[reduced_vectors[i][1]],
                        z=[reduced_vectors[i][2]],
                        mode='markers+text',
                        marker=dict(size=6, color='skyblue'),
                        name=tickers[i],
                        text=[tickers[i]],
                        textposition='top center'
                    ))

                fig.update_layout(
                    title='3D Visualization of Similar Stocks',
                    scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'
                    ),
                    height=600
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No recommendations found.")
        else:
            st.error("Failed to prepare data. See logs for more details.")

elif page == "🚨 Anomaly Detection":
    st.title("🚨 Stock Anomaly Explorer")

    stock_symbol = st.text_input("Enter Stock Symbol:", value="NVDA")
    col1, col2 = st.columns(2)
    with col1:
        from_date = st.date_input("From Date:", value=datetime(2023, 1, 1))
    with col2:
        to_date = st.date_input("To Date:", value=datetime(2025, 4, 17))

    if st.button("Explore Anomalies"):
        with st.spinner("Hunting for anomalies..."):
            try:
                df_with_returns, fig = anomaly_detection(
                    stock_symbol.upper(),
                    start_date=from_date.strftime("%Y-%m-%d"),
                    end_date=to_date.strftime("%Y-%m-%d")
                )
                if fig:
                    st.pyplot(fig)

                    st.info("💡 Hint: Red 'x' marks indicate days with unusually large price changes compared to the typical behavior of this stock.")
                    
                    if not df_with_returns.empty:
                        df_with_returns['Returns'] = df_with_returns['Returns'] * 100
                        df_with_returns.index = pd.to_datetime(df_with_returns.index).date  
                        df_with_returns.index.name = "Date" 
                        df_with_returns = df_with_returns.reset_index()
                        st.write("### Anomalies and their returns")
                        st.dataframe(df_with_returns)
                    else:
                        st.info("No anomalies detected.")
                else:
                    st.error("Anomaly detection failed. Please check the stock symbol and date range.")
            except Exception as e:
                st.error(f"Error during anomaly detection: {e}")