import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class DataLoader:
    prices: pd.DataFrame 
    scaler: MinMaxScaler
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_size: int
    lookback: int

def get_prices(ticker: str, start_date: str, end_date: Optional[str]=None, visualize_prices: bool=False) -> pd.DataFrame:
    """
    Loads data from Yahoo Finance API using yfinance
    
    Parameters:
        ticker (str):
            Unique short code that identifies a stock.
        start_date (str):
            First day of stock to be loaded.
        end_date (Optional[str]):
            Last day of stock to be loaded.
        visualize_prices (bool):
            Enables/disables a graph to visualize the data.
    
    Returns:
        df (pd.DataFrame):
            Data frame of the closing prices of the stock.
    """
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for ticker={ticker}.")
    
    # Only use close prices and sort df by date
    df = df[["Close"]].dropna()
    df = df.sort_values('Date')
    print(f'Loaded {ticker} data from Yahoo Finance successfully!')

    # Data visualization
    if visualize_prices:
        plt.figure()
        plt.plot(df.index, df['Close'])
        plt.title(f'{ticker} Closing Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.show()
    
    return df

def _create_sequences(scaled_data: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Creates LSTM sequences by converting prices to supervised learning samples, guaranteeing proper temporal structure.
    
    Parameters:
        scaled_data (np.ndarray): 
            Contains scaled train and test data.
            The array is of the shape (N,1)
        lookback (int): 
            Number of days for model to base predictions on.

    Returns:
        X (np.ndarray):
            Contains prices from lookback number of days.
            The array is of the shape (N-lookback, lookback, 1)
        y (np.ndarray):
            Contains the target for a prediction at day i.
            The array is of the shape (N-lookback, 1)
    """
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback : i])
        y.append(scaled_data[i])
    
    return np.array(X), np.array(y)

def prepare_dataset(prices: pd.DataFrame, lookback: int, train_split: float) -> DataLoader:
    """
    Splits the data into a training set and a testing set.
    Normalizes both test and train data with respect to train data.
    
    Parameters:
        prices (pd.DataFrame):
            Contains price data of a particular stock.
        lookback (int):
            Number of days for model to base predictions on.
        train_split (float):
            The percentage of data belonging to the training dataset.
            Must be between 0 and 1.
    
    Returns:
        DataLoader (object):
            Contains all necessary attributes to train LSTM model.
    """
    if not (0.0 < train_split < 1.0):
        raise ValueError('train_split must be between 0 and 1.')
    
    train_size = int(len(prices) * train_split)
    if train_size <= lookback:
        raise ValueError('Train set is too small for the chosen lookback')
    
    train_data = prices.iloc[:train_size]
    test_data = prices.iloc[train_size:]

    # Fit scaler exclusively on training data to avoid data leakage and ensure that
    # test data is normalized using only information available at training time.
    scaler = MinMaxScaler(feature_range=(0,1))
    train_data = scaler.fit_transform(train_data.values)
    test_data = scaler.transform(test_data.values)

    # Combine train + test for continuity as 
    # test window needs training data for historical context.
    # Build sequences on combined series so test windows include
    # context from train data
    full_scaled_data = np.vstack((train_data, test_data))
    X_all, y_all = _create_sequences(full_scaled_data, lookback)

    # Split sequences into train and test
    train_end = train_size - lookback

    X_train = X_all[:train_end]
    y_train = y_all[:train_end]

    X_test = X_all[train_end:]
    y_test = y_all[train_end:]

    return DataLoader(
        prices=prices,
        scaler=scaler,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_size=train_size,
        lookback=lookback
    )
