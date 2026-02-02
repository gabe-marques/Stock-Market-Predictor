# Stock-Market-Predictor

This project implements a long short-term memory (LSTM) neural network to analyze and predict stock prices using historical market data. It is intended for educational and exploratory purposes.

This project is a **modernized rewrite** of a legacy LSTM stock market tutorial originally written for TensorFlow 1.x, updated for current tooling, APIs, and reproducibility standards.

---

## Disclaimer

This project is intended for educational purposes only.
It is **not financial advice** and should not be used to make real investment decisions.

Stock market predictions are inherently uncertain, and past performance does not guarantee future results. 

This model predicts prices, not trading signals. The model does not take into account external factors such as news. Thus, results should **NOT** be used for financial decision-making.

---

## Project Overview

Stock prices are sequential time-series data, peferct for recurrent neural networks. In this project, I:

- Download historical stock data using `yfinance`
- Normalize the data using training-only statistics to prevent data leakage
- Convert the time series into supervised learning sequences
- Train an LSTM-based regression model

---

## Model Arhictecture

The model consists of:

- Two stacked LSTM layers
- Dropout regularization to reduce overfitting 
- A fully connected output layer for price prediction

The model is trained using the Adam optimizer and Mean Squared Error (MSE) loss.

---

## Data Processing Pipeline

1. Data Collection
- Historical stock prices are downloaded using `yfinance`
- Only the Close price is used for modeling 

2. Train/Test Split
- The dataset is split chronologically (80% train, 20% test)
- No shuffling is performed to preserve temporal order 

3. Normalization
- A `MinMaxScaler` is fit only on training data
- Test data is transformed using the same scaler to avoid data leakage

4. Sequence Generation
- Sliding window of fixed length (default: 60 days)
- Each window predicts the next closing price

---

## Project Structure

```text
project/
├── configs/
│   └── config.json          # Data, model, and training configuration
├── src/
│   ├── data.py              # Data loading and preprocessing
│   ├── model.py             # LSTM model definition
│   └── train.py             # Training and evaluation script
├── requirements.txt
└── README.md
```

---

## Configuration

Project parameters are stored in a JSON config file (`configs/config.json`), including:

- Stock ticker
- Date range
- Lookback window size
- Train/test split
- Training hyperparameters

This allows experiments to be run without modifying source code

---

## Installation

Requirements:

- Python 3.10
- Tensorflow >= 2.10

Install dependencies:

`pip install -r requirements.txt`

Run Training:

`python -m src.train --config configs/config.json`

---

## Evaluation Metrics

Model performance is evaluated using:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

Theese metrics measure how closely predicted prices match actual prices on the test set.

---

## Visualization

The final output includes a plot comparing:
- Actual closing prices
- LSTM-predicted prices

This provides a qualitative view of how well the model captures overall trends.

---

## Future Improvements

Possible extensions include:

- Predicting returns or direction instead of prices
- Adding technical indicators to improve predictions (RSI, MACD)
- Multi-stock training
- Hyperparameter tuning

---

## Notes

This project modernizes an LSTM stock prediction tutorial originally written for TensorFlow 1.6 by refactoring it to TensorFlow 2.x and Keras APIs. Built using Python 3.10.

Original DataCamp LSTM stock market tutorial (TensorFlow 1.6) https://www.datacamp.com/tutorial/lstm-python-stock-market.

Improvements coming soon!
