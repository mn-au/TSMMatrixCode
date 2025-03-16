# Import necessary libraries for financial data, numerical computations, and plotting
import yfinance as yf                   # To fetch historical financial data
import numpy as np                      # For numerical operations
import pandas as pd                     # For data manipulation and analysis
import pywt                           # For performing wavelet transforms
import matplotlib.pyplot as plt         # For plotting graphs

# Import machine learning modules for dimensionality reduction, modeling, and evaluation
from sklearn.decomposition import PCA  # Principal Component Analysis for dimensionality reduction
from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier for predictive modeling
from sklearn.preprocessing import StandardScaler  # To standardize features
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV  # Time series CV and grid search for hyperparameter tuning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Evaluation metrics

# ------------------------------------------------------------------------
# 1. Data Fetcher & Preprocessor
# ------------------------------------------------------------------------

# Class to fetch historical data for multiple tickers and perform basic preprocessing
class DataFetcher:
    """
    Fetches historical price/volume data for multiple tickers via yfinance,
    along with basic preprocessing (handling missing values, etc.).
    """
    def __init__(self, tickers, period="10y", interval="1d"):
        # Initialize with list of tickers, time period, and data interval
        self.tickers = tickers
        self.period = period
        self.interval = interval
    
    def fetch_all(self):
        # Dictionary to hold fetched data for each ticker
        data_dict = {}
        for ticker in self.tickers:
            # Fetch historical data for the given ticker
            df = yf.Ticker(ticker).history(period=self.period, interval=self.interval)
            # Keep only essential columns: Open, High, Low, Close, Volume
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            # Drop any rows with missing values
            df.dropna(inplace=True)
            # Store the processed DataFrame in the dictionary
            data_dict[ticker] = df
        # Return the dictionary containing data for all tickers
        return data_dict

# Class to preprocess the fetched data by computing returns and rolling volatility
class DataPreprocessor:
    """
    Computes returns (log or simple) and rolling volatility.
    """
    def __init__(self, use_log_returns=False):
        # Flag to decide between log returns and simple percentage returns
        self.use_log_returns = use_log_returns
    
    def compute_returns(self, df):
        """
        Adds a 'Return' column: log returns or simple returns.
        """
        df = df.copy()  # Work on a copy of the DataFrame
        if self.use_log_returns:
            # Compute log returns using the natural log of the ratio of consecutive closing prices
            df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
        else:
            # Compute simple returns as the percentage change of closing prices
            df['Return'] = df['Close'].pct_change()
        # Remove rows with NaN values (resulting from shifting)
        df.dropna(inplace=True)
        return df
    
    def compute_volatility(self, df, window=20):
        """
        Adds a 'Volatility' column: rolling std dev of 'Return' over 'window' days.
        """
        df = df.copy()  # Work on a copy of the DataFrame
        # Compute rolling standard deviation of returns to represent volatility
        df['Volatility'] = df['Return'].rolling(window).std()
        # Drop any rows with missing values after rolling calculation
        df.dropna(inplace=True)
        return df
    
    def process_data(self, df, vol_window=20):
        # First compute returns, then compute volatility
        df = self.compute_returns(df)
        df = self.compute_volatility(df, window=vol_window)
        return df

# ------------------------------------------------------------------------
# 2. Feature Engineering (TSM-Inspired) + Noise Modeling
# ------------------------------------------------------------------------

# Class to create advanced features based on wavelet transforms, Fourier transforms, etc.
class FeatureEngineer:
    """
    Creates advanced features:
      - Wavelet decomposition (multi-scale),
      - (Optional) wavelet-based noise modeling via thresholding,
      - Rolling volatility & volume spikes,
      - Fourier transform of recent returns,
      - (Optionally) PCA can be applied after scaling in the pipeline.
    """
    def __init__(self, wavelet='db4', wavelet_level=3, fft_coefs=5, window_size=30,
                 use_noise_threshold=False, noise_threshold=0.01):
        """
        wavelet        : wavelet family (e.g. 'db4')
        wavelet_level  : wavelet decomposition level
        fft_coefs      : number of real & imag FFT coefficients to keep
        window_size    : length of rolling window for feature extraction
        use_noise_threshold : whether to apply wavelet thresholding for noise modeling
        noise_threshold     : threshold for wavelet coefficient denoising
        """
        self.wavelet = wavelet
        self.wavelet_level = wavelet_level
        self.fft_coefs = fft_coefs
        self.window_size = window_size
        self.use_noise_threshold = use_noise_threshold
        self.noise_threshold = noise_threshold
    
    def wavelet_decomposition(self, series):
        """
        Decompose the series into wavelet coefficients. If use_noise_threshold is True,
        apply soft thresholding for noise modeling.
        """
        # Perform wavelet decomposition on the input series
        coeffs = pywt.wavedec(series, self.wavelet, level=self.wavelet_level)
        if self.use_noise_threshold:
            # Loop through detail coefficients (skip approximation at index 0)
            for i in range(1, len(coeffs)):
                # Apply soft thresholding to reduce noise in detail coefficients
                coeffs[i] = pywt.threshold(coeffs[i], self.noise_threshold, mode='soft')
        # Flatten all wavelet coefficients into a single feature vector
        features = np.hstack(coeffs)
        return features
    
    def generate_features(self, df):
        """
        For each row after 'window_size' days, compute wavelet decomposition,
        rolling volatility, volume spike, Fourier transform.
        """
        df = df.copy()  # Create a copy to avoid modifying original data
        feature_list = []  # List to store feature vectors
        index_list = []    # List to store corresponding indices (dates)

        # Loop over data starting from 'window_size' to ensure enough past data for features
        for i in range(self.window_size, len(df)):
            # Extract a window of returns for wavelet and Fourier transforms
            ret_window = df['Return'].iloc[i-self.window_size:i].values
            # Extract a window of volume data to compute volume spikes
            vol_window = df['Volume'].iloc[i-self.window_size:i].values
            
            # 1) Wavelet decomposition (and noise modeling if enabled)
            wavelet_features = self.wavelet_decomposition(ret_window)
            
            # 2) Rolling volatility is taken as the current day's volatility value
            rolling_vol = df['Volatility'].iloc[i]
            
            # 3) Compute volume spike as the ratio of current volume to average volume of the window
            avg_vol = np.mean(vol_window)
            current_vol = df['Volume'].iloc[i]
            vol_spike = current_vol / avg_vol if avg_vol != 0 else 1.0
            
            # 4) Fourier transform on the return window
            fft_vals = np.fft.fft(ret_window)
            # Keep first few real parts
            fft_real = fft_vals.real[:self.fft_coefs]
            # Keep first few imaginary parts
            fft_imag = fft_vals.imag[:self.fft_coefs]
            
            # Combine all features into a single vector
            combined_features = np.hstack([
                wavelet_features,
                rolling_vol,
                vol_spike,
                fft_real,
                fft_imag
            ])
            
            # Append feature vector and corresponding index
            feature_list.append(combined_features)
            index_list.append(df.index[i])
        
        # Create a DataFrame from the feature list with corresponding date indices
        feature_df = pd.DataFrame(feature_list, index=index_list)
        return feature_df

# ------------------------------------------------------------------------
# 3. Target Creation (Momentum Labels)
# ------------------------------------------------------------------------

# Class to label each day as bullish or bearish based on future cumulative returns
class TargetCreator:
    """
    Labels each day as bullish/bearish based on the next 'lookahead' days' cumulative return.
    """
    def __init__(self, lookahead=5, threshold=0.0):
        # Number of days ahead to look for determining momentum and the return threshold for bullish label
        self.lookahead = lookahead
        self.threshold = threshold
    
    def create_labels(self, df):
        df = df.copy()  # Work on a copy of the DataFrame
        returns = df['Return'].values  # Get array of returns
        targets = []  # List to store binary labels
        # Loop over returns, ensuring not to go out of bounds by stopping lookahead days early
        for i in range(len(returns) - self.lookahead):
            # Compute cumulative return over the lookahead period
            future_return = np.prod(1 + returns[i+1:i+1+self.lookahead]) - 1
            # Label as 1 (bullish) if cumulative return exceeds threshold, else 0 (bearish)
            label = 1 if future_return > self.threshold else 0
            targets.append(label)
        return np.array(targets)

# ------------------------------------------------------------------------
# 4. Model Manager (TimeSeries CV + Random Forest)
# ------------------------------------------------------------------------

# Class to handle model training, evaluation, and diagnostics using a Random Forest
class ModelManager:
    """
    Uses a Random Forest with TimeSeriesSplit for hyperparameter tuning,
    then evaluates with classification metrics.
    """
    def __init__(self, n_splits=5):
        # Number of splits for time series cross-validation and placeholder for the trained model
        self.n_splits = n_splits
        self.model = None
    
    def train_random_forest(self, X_train, y_train):
        """
        GridSearch over a small param grid, using TimeSeriesSplit for CV.
        """
        # Define grid of hyperparameters for the Random Forest
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5]
        }
        # Instantiate a RandomForestClassifier with a fixed random state for reproducibility
        rf = RandomForestClassifier(random_state=42)
        # Setup time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        # GridSearchCV to find the best hyperparameters using time series CV
        grid_search = GridSearchCV(rf, param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        
        # Print best found parameters
        print("Best parameters found:", grid_search.best_params_)
        # Save the best estimator as the model to use
        self.model = grid_search.best_estimator_
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Prints accuracy, classification report, confusion matrix.
        Returns predicted labels.
        """
        if not self.model:
            # Raise error if model has not been trained yet
            raise ValueError("No model has been trained yet.")
        
        # Predict labels for the test set
        y_pred = self.model.predict(X_test)
        # Calculate accuracy score
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {acc*100:.2f}%")
        
        # Print detailed classification report (precision, recall, f1-score)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Print confusion matrix showing true vs predicted labels
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        return y_pred
    
    def plot_diagnostics(self, X_test, y_test, test_index):
        """
        Plot predicted probabilities over time, comparing them to actual labels.
        """
        if not self.model:
            # Raise error if model is not trained
            raise ValueError("No model has been trained yet.")
        
        # Get predicted probabilities for the positive class (bullish)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        # Create a DataFrame to hold predictions and actual labels with time index
        diag_df = pd.DataFrame({
            'Predicted_Probability_Bullish': y_prob,
            'Actual_Label': y_test
        }, index=test_index)
        
        # Plot the predicted probabilities and actual labels
        plt.figure(figsize=(12, 6))
        plt.plot(diag_df.index, diag_df['Predicted_Probability_Bullish'], label='Predicted Bullish Probability', color='blue')
        plt.scatter(diag_df.index, diag_df['Actual_Label'], label='Actual Label (0 or 1)', color='red', marker='o', alpha=0.5)
        plt.title("Momentum Strength Indicator Over Time")
        plt.xlabel("Time")
        plt.ylabel("Probability / Label")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Print statistics of predicted probabilities
        print("Prediction Confidence Statistics:")
        print(diag_df['Predicted_Probability_Bullish'].describe())
        
        return pd.Series(y_prob, index=test_index)

# ------------------------------------------------------------------------
# 5. MomentumPipeline (with optional PCA & wavelet noise modeling)
# ------------------------------------------------------------------------

# Main pipeline class that integrates data fetching, preprocessing, feature engineering,
# target creation, model training/evaluation, and diagnostics.
class MomentumPipeline:
    """
    Main pipeline:
      1) Fetch data
      2) Preprocess (returns + volatility)
      3) Feature Engineering (wavelet, Fourier, volume spike, etc.)
         + optional wavelet-based noise modeling
      4) (Optional) PCA after scaling
      5) Train & Evaluate Random Forest with time-series CV
      6) Chronological final train/test split to avoid data leakage
    """
    def __init__(self, tickers, period="10y", interval="1d",
                 use_log_returns=False,
                 wavelet='db4', wavelet_level=3, fft_coefs=5, window_size=30,
                 use_noise_threshold=False, noise_threshold=0.01,
                 use_pca=False, pca_components=10,
                 lookahead=5, threshold=0.0, test_size=0.2):
        
        # Basic settings for tickers, data period, interval, and test split size
        self.tickers = tickers
        self.period = period
        self.interval = interval
        self.test_size = test_size
        
        # Initialize the DataFetcher with tickers and date parameters
        self.data_fetcher = DataFetcher(tickers=self.tickers, period=self.period, interval=self.interval)
        # Initialize the DataPreprocessor with log returns flag
        self.preprocessor = DataPreprocessor(use_log_returns=use_log_returns)
        
        # Initialize FeatureEngineer with wavelet and feature extraction settings
        self.feature_engineer = FeatureEngineer(
            wavelet=wavelet,
            wavelet_level=wavelet_level,
            fft_coefs=fft_coefs,
            window_size=window_size,
            use_noise_threshold=use_noise_threshold,
            noise_threshold=noise_threshold
        )
        
        # Initialize TargetCreator with lookahead window and threshold
        self.target_creator = TargetCreator(lookahead=lookahead, threshold=threshold)
        
        # Initialize ModelManager for model training and evaluation
        self.model_manager = ModelManager(n_splits=5)
        
        # PCA settings: whether to use PCA and number of principal components
        self.use_pca = use_pca
        self.pca_components = pca_components
    
    def run_pipeline(self):
        # Fetch data for all tickers using the DataFetcher
        data_dict = self.data_fetcher.fetch_all()
        
        # Loop through each ticker's data
        for ticker, df in data_dict.items():
            print("\n" + "="*60)
            print(f"Processing Ticker: {ticker}")
            print("="*60)
            
            # 1) Preprocess the data (compute returns and volatility)
            df_proc = self.preprocessor.process_data(df, vol_window=20)
            # Skip ticker if there is insufficient data after preprocessing
            if len(df_proc) < 50:
                print(f"Not enough data after preprocessing for {ticker}. Skipping...")
                continue
            
            # 2) Generate features using the FeatureEngineer
            feature_df = self.feature_engineer.generate_features(df_proc)
            # Skip ticker if no features were generated
            if feature_df.empty:
                print(f"No features generated for {ticker}. Skipping...")
                continue
            
            # 3) Create target labels based on future cumulative returns
            targets = self.target_creator.create_labels(df_proc)
            # Align feature DataFrame and targets by removing extra rows from the end
            feature_df = feature_df.iloc[:-self.target_creator.lookahead] if len(feature_df) > self.target_creator.lookahead else feature_df
            targets = targets[:len(feature_df)]
            
            # Skip ticker if the aligned dataset is too small
            if len(feature_df) < 50:
                print(f"Insufficient data points for {ticker} after alignment. Skipping...")
                continue
            
            # 4) Scale the features using StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(feature_df.values)
            
            # If PCA is enabled, reduce the dimensionality of the features
            if self.use_pca:
                pca = PCA(n_components=self.pca_components)
                X_pca = pca.fit_transform(X_scaled)
                X = X_pca
                print(f"PCA applied: original dim={X_scaled.shape[1]}, reduced dim={X_pca.shape[1]}")
            else:
                X = X_scaled
            
            # Set y as the target labels
            y = targets
            
            # Final chronological train/test split to avoid data leakage
            split_index = int(len(X) * (1 - self.test_size))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            test_index = feature_df.index[split_index:]
            
            # 5) Train and evaluate the Random Forest model
            print("Training model for", ticker)
            self.model_manager.train_random_forest(X_train, y_train)
            
            print("Evaluating model for", ticker)
            y_pred = self.model_manager.evaluate_model(X_test, y_test)
            
            # 6) Plot diagnostic charts comparing predicted probabilities and actual labels
            print("Plotting diagnostic charts for", ticker)
            _ = self.model_manager.plot_diagnostics(X_test, y_test, test_index)
            
            # Plot feature importances from the trained model
            importances = self.model_manager.model.feature_importances_
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(importances)), importances)
            plt.title(f"Feature Importances - {ticker}")
            plt.xlabel("Feature Index")
            plt.ylabel("Importance")
            plt.show()

# ------------------------------------------------------------------------
# 6. Usage Example
# ------------------------------------------------------------------------

# If this script is run as the main program, execute the following example pipeline
if __name__ == "__main__":
    """
    A TSM-inspired momentum detection pipeline with:
      - Wavelet transforms + optional noise thresholding
      - Rolling volatility & volume spike
      - Fourier transforms
      - (Optional) PCA for dimensionality reduction
      - TimeSeriesSplit for hyperparam tuning, chronological final test

    Modify constructor arguments to experiment with:
      - wavelet_level
      - use_noise_threshold, noise_threshold
      - use_pca, pca_components
      - etc.
    """
    # Define a list of tickers to test; here, using SPY as an example
    tickers_to_test = ["SPY"] 

    # Initialize the MomentumPipeline with specified parameters
    pipeline = MomentumPipeline(
        tickers=tickers_to_test,
        period="10y",
        interval="1d",
        use_log_returns=False,      # or True for log returns
        wavelet='db4',
        wavelet_level=1,           # wavelet decomposition level
        fft_coefs=5,
        window_size=30,
        use_noise_threshold=True,   # Enable wavelet-based noise modeling
        noise_threshold=0.01,       # Set threshold for denoising wavelet details
        use_pca=True,               # Enable PCA for dimensionality reduction
        pca_components=10,          # Number of principal components to keep
        lookahead=5,
        threshold=0.0,
        test_size=0.2               # 20% of data reserved for testing
    )
    
    # Run the full pipeline: fetching data, feature engineering, model training/evaluation, and diagnostics
    pipeline.run_pipeline()
