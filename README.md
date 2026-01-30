# Stock-Market-Predictor

This project implements a long short-term memory (LSTM) neural network to analyze and predict stock prices using historical market data. It is intended for educational and exploratory purposes.

This project is a **modernized rewrite** of a legacy LSTM stock market tutorial originally written for TensorFlow 1.x, updated for current tooling, APIs, and reproducibility standards.

---

## Disclaimer

This project is intended for educational purposes only.
It is **not financial advice** and should not be used to make real investment decisions.

Stock market predictions are inherently uncertain, and past performance does not guarantee future results. 

Use this project at your own risk.

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

## Future Improvements

Possible extensions include:

- Predicting returns or direction instead of prices
- Adding technical indicators to improve predictions (RSI, MACD)
- Multi-stock training
- Hyperparameter tuning

---

## Notes

This project follows a guide from https://www.datacamp.com/tutorial/lstm-python-stock-market.

This project modernizes an LSTM stock prediction tutorial originally written for TensorFlow 1.6 by refactoring it to TensorFlow 2.x and Keras APIs. Built using Python 3.10.

Improvements coming soon!
