# Momentum Detection Pipeline (TSM-Inspired)

This project implements a momentum detection system for financial markets using machine learning, signal processing, and time-series modeling. It leverages wavelet transforms, Fourier transforms, PCA, rolling volatility, and volume spike detection to extract and analyze features from historical stock data.

## Features & Workflow

1. **Data Fetching & Preprocessing**  
   - Retrieves historical price/volume data from Yahoo Finance using `yfinance`.  
   - Computes returns (log or simple) and rolling volatility.

2. **Feature Engineering & Noise Modeling**  
   - Applies wavelet decomposition with optional noise thresholding.  
   - Extracts Fourier coefficients and calculates volume spike metrics.

3. **Target Creation**  
   - Generates momentum labels (bullish or bearish) based on future cumulative returns.

4. **Model Training & Evaluation**  
   - Uses a Random Forest Classifier with TimeSeriesSplit and GridSearchCV for hyperparameter tuning.  
   - Evaluates performance with accuracy, classification report, and diagnostic plots.

5. **Pipeline Integration**  
   - The `MomentumPipeline` class orchestrates the entire process from data fetching to model diagnostics.

## Usage 

```python
if __name__ == "__main__":
    tickers_to_test = ["SPY"]
    pipeline = MomentumPipeline(
        tickers=tickers_to_test,
        period="10y",
        interval="1d",
        use_log_returns=False,
        wavelet='db4',
        wavelet_level=1,
        fft_coefs=5,
        window_size=30,
        use_noise_threshold=True,
        noise_threshold=0.01,
        use_pca=True,
        pca_components=10,
        lookahead=5,
        threshold=0.0,
        test_size=0.2
    )
    pipeline.run_pipeline()
