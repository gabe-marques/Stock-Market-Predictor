import json
from pathlib import Path
import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Hide info and warning messages

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

from src.data_loader import get_prices, prepare_dataset
from src.model import build_lstm_model

def load_config(path: str|Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Config not found: {path}')
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)
    
def main(config_path: str) -> None:
    configs = load_config(config_path)
    
    # Data configs
    ticker = configs['data']['ticker']
    start_date = configs['data']['start_date']
    end_date = configs['data']['end_date']

    # Model configs
    train_split = configs['model']['train_split']
    lookback = configs['model']['lookback']
    units1 = configs['model']['units1']
    units2 = configs['model']['units2']
    dropout = configs['model']['dropout']
    learning_rate = configs['model']['learning_rate']

    # Training configs
    epochs = configs['training']['epochs']
    batch_size = configs['training']['batch_size']
    patience = configs['training']['early_stopping_patience']
    val_split = configs['training']['validation_split']

    output_dir = Path(configs['output']['folder_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Configs successfully loaded from {config_path}!')

    # Set seed for reproducibility 
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load data
    prices = get_prices(ticker, start_date, end_date)
    input_data = prepare_dataset(prices, lookback, train_split)

    # Build model 
    model = build_lstm_model(lookback, units1, units2, dropout, learning_rate)
    print(model.summary())

    # Train model
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    history = model.fit(
        input_data.X_train,
        input_data.y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    # Predict
    y_pred_scaled = model.predict(input_data.X_test, verbose=0)
    y_test_inv = input_data.scaler.inverse_transform(input_data.y_test)
    y_pred_inv = input_data.scaler.inverse_transform(y_pred_scaled)

    # Metrics
    rmse = float(np.sqrt(mean_squared_error(y_test_inv, y_pred_inv)))
    mae = float(mean_absolute_error(y_test_inv, y_pred_inv))
    print('\nMetrics:')
    print(f'Ticker: {ticker}')
    print(f'RMSE: {rmse:.4f}')
    print(f'MAE : {mae:.4f}')

    # Visualize prediction
    pred_index = input_data.prices.index[input_data.train_size:]
    plt.figure()
    plt.plot(input_data.prices.index, input_data.prices["Close"].values, label="Actual Close")
    plt.plot(pred_index, y_pred_inv.flatten(), label="Predicted Close")
    plt.title(f'{ticker} Close Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    fig_path = output_dir / f'{ticker}_prediction_plot.png'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.show()

    # Save outputs
    model_path = output_dir / f'{ticker}_lstm.keras'
    scaler_path = output_dir / f'{ticker}_scaler.joblib'
    history_path = output_dir / f'{ticker}_history.json'

    model.save(model_path)
    joblib.dump(input_data.scaler, scaler_path)
    with history_path.open('w', encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    print(f"\nSaved model  -> {model_path}")
    print(f"Saved scaler -> {scaler_path}")
    print(f"Saved history-> {history_path}")
    print(f"Saved plot   -> {fig_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.json', help='Path to JSON config.')
    args = parser.parse_args()
    main(args.config)