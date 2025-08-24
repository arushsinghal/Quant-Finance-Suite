# StockSense 

Welcome to **StockSense**, a comprehensive web application designed to provide insights into stock market trends, predict future prices, recommend similar stocks, and detect anomalies in stock behavior.


## Features 

### 1. **Stock Price Prediction**
- Predict the next day's closing price for a selected stock using **LSTM (Long Short-Term Memory)** neural networks.
- Evaluate the model's performance with metrics like:
  - **MAPE (Mean Absolute Percentage Error)**
  - **R² Score**
  - **NRMSE (Normalized Root Mean Squared Error)**

### 2. **Similar Stock Recommendations**
- Find stocks similar to a selected stock based on various indicators using **K nearest neighbors**.
- Filter recommendations by sector (e.g., Technology, Healthcare).
- Visualize recommendations in a **3D plot using PCA (Principal Component Analysis)** for better visualisation.

### 3. **Anomaly Detection**
- Detect unusual stock price movements using the **Isolation Forest** algorithm.
- Highlight anomalies in daily returns with a clear visualization as well as in a tabular format for detailed analysis.

### 4. **Caching for Improved Performance**
- Leverages Streamlit's inbuilt caching to cache feature vectors and price prediction models for **1 day** (24 hours), ensuring faster performance and reduced computation time for repeated operations.


## Installation 

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Armaan457/StockSense.git
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   \env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```


## Technologies used 
- **AI/ML**: Tensorflow and scikit-learn
- **Data**: yfinance and pandas
- **Visualizations**: Matplotlib and plotly
- **User Interface**: Streamlit


## Developers
- Armaan Jagirdar
- Arnav Aggarwal
