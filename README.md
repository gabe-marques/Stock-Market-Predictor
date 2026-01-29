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

## Project Structure

.
├── configs/
│   └── config.json          # Data, model, and training configuration
├── src/
│   ├── data.py              # Data loading and preprocessing
│   ├── model.py             # LSTM model definition
│   └── train.py             # Training and evaluation script
├── requirements.txt
└── README.md

---

## Installation
Steps coming soon!

---

## Features (Current)
- Loads and visualizes historical stock price data

---

## Planned Features
- Prepares data for LSTM modeling 
- Trains a TensorFlow LSTM model
- Visualizes actual vs predicted prices

---

## Notes

This project follows a guide from https://www.datacamp.com/tutorial/lstm-python-stock-market.

This project modernizes an LSTM stock prediction tutorial originally written for TensorFlow 1.6 by refactoring it to TensorFlow 2.x and Keras APIs. Built using Python 3.10.

Improvements coming soon!
