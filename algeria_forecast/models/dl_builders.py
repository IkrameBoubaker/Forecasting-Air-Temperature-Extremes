"""Keras model architectures (ANN, LSTM, CNN-LSTM) and shared training callbacks.

All three models take lag-only input (no auxiliary seasonal/volatility
features): ANN takes the flat (lag,) vector, LSTM and CNN-LSTM take
the (lag, 1) sequence through their recurrent/convolutional layers
directly into the final dense/output layers.
"""

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout,
    Conv1D, MaxPooling1D,
    Input,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_ann(n_features, hidden_units, dropout, lr):
    inp = Input(shape=(n_features,), name="ann_input")
    x = Dense(hidden_units, activation="relu")(inp)
    x = Dropout(dropout)(x)
    x = Dense(max(hidden_units // 2, 16), activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1, name="ann_output")(x)
    m = Model(inp, out, name="ANN")
    m.compile(optimizer=Adam(lr), loss="mse")
    return m


def build_lstm(lag, units, dropout, lr):
    seq_in = Input(shape=(lag, 1), name="lag_input")
    x = LSTM(units, return_sequences=True)(seq_in)
    x = LSTM(max(units // 2, 16))(x)
    x = Dropout(dropout)(x)

    d = Dense(32, activation="relu")(x)
    out = Dense(1)(d)

    m = Model(seq_in, out, name="LSTM")
    m.compile(optimizer=Adam(lr), loss="mse")
    return m


def build_cnn_lstm(lag, filters, lstm_units, kernel_size, dropout, lr):
    seq_in = Input(shape=(lag, 1), name="lag_input")
    x = Conv1D(filters, kernel_size=kernel_size, padding="same",
                activation="relu")(seq_in)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters * 2, kernel_size=kernel_size, padding="same",
                activation="relu")(x)
    x = LSTM(lstm_units)(x)
    x = Dropout(dropout)(x)

    d = Dense(32, activation="relu")(x)
    out = Dense(1)(d)

    m = Model(seq_in, out, name="CNN_LSTM")
    m.compile(optimizer=Adam(lr), loss="mse")
    return m


def dl_callbacks(patience: int):
    return [
        EarlyStopping(monitor="val_loss", patience=patience,
                       restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                            patience=7, min_lr=1e-6, verbose=0),
    ]
