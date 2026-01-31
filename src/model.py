import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_model(lookback:  int, units1: int, units2: int, dropout: float, learning_rate) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(lookback, 1)), # X data is shaped like (num_samples, lookback, 1) num_sample = number of windows, lookback = timesteps per window, 1 = number of features per timestep (closing price)
            layers.LSTM(units1, return_sequences=True), # First LSTM layer learns short-term local patterns, units1 controls how much the model can learn, return_sequence allows another LSTM layer to be stacked since LSTM expects sequence as input
            layers.Dropout(dropout), # Helps prevent memorization of training data
            layers.LSTM(units2), # Second LSTM layer aggregates patterns to learn longer term trends, return_sequence is false by default, feed into final regression layer
            layers.Dropout(dropout),
            layers.Dense(1),  # Outputs one value: the prediction of the close price for the next day
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model