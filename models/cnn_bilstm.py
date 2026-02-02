import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Add, Dropout, Dense,
    Bidirectional, LSTM, LayerNormalization,
    GlobalAveragePooling1D, MultiHeadAttention
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


# ======================================================
# 0. Reproducibility (optional)
# ======================================================
def set_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ======================================================
# 1A. Build supervised sequence data (1-step, backward compatible)
# ======================================================
def build_sequences(X, y, seq_len=24):
    """
    1-step: X[t-seq_len:t] -> y[t]
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)


# ======================================================
# 1B. Build supervised sequence data (multi-step, paper-like)
# ======================================================
def build_sequences_multi(X, y, input_len=48, horizon=24):
    """
    Multi-step (paper-like):
      X[t-input_len:t] -> y[t : t+horizon]
    Shapes:
      X_seq: (n_samples, input_len, n_features)
      y_seq: (n_samples, horizon)
    """
    X_seq, y_seq = [], []
    n = len(X)
    end = n - horizon  # last t such that y[t:t+horizon] exists
    for t in range(input_len, end):
        X_seq.append(X[t - input_len:t])
        y_seq.append(y[t:t + horizon])
    return np.array(X_seq), np.array(y_seq)


# ======================================================
# 2. Model building blocks
# ======================================================
def _residual_cnn_block(x, filters: int, kernel_size: int, dilation: int, dropout: float):
    shortcut = x

    x = LayerNormalization()(x)
    x = Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation, activation="relu")(x)

    x = LayerNormalization()(x)
    x = Dropout(dropout)(x)
    x = Conv1D(filters, 1, padding="same")(x)

    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding="same")(shortcut)

    x = Add()([x, shortcut])
    x = tf.keras.layers.Activation("relu")(x)
    return x


# ======================================================
# 3. CNN-BiLSTM(+Attention) model (supports horizon)
# ======================================================
def build_cnn_bilstm_model(
    seq_len: int,
    n_features: int,
    horizon: int = 1,                 # ✅ NEW: 1-step or multi-step
    learning_rate: float = 3e-4,
    cnn_dropout: float = 0.15,
    lstm_dropout: float = 0.2,
    lstm_units: int = 128,
    use_attention: bool = True,
    huber_delta: float = 1.0,
    clipnorm: float = 1.0
):
    """
    If horizon == 1: output shape (batch, 1)
    If horizon  > 1: output shape (batch, horizon) -> multi-step forecasting
    """
    inp = Input(shape=(seq_len, n_features))

    x = Conv1D(64, 3, padding="same", activation="relu")(inp)
    x = _residual_cnn_block(x, filters=64, kernel_size=3, dilation=1, dropout=cnn_dropout)
    x = _residual_cnn_block(x, filters=64, kernel_size=3, dilation=2, dropout=cnn_dropout)
    x = _residual_cnn_block(x, filters=128, kernel_size=3, dilation=4, dropout=cnn_dropout)

    x = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=lstm_dropout))(x)
    x = Bidirectional(LSTM(lstm_units // 2, return_sequences=True, dropout=lstm_dropout))(x)

    if use_attention:
        attn = MultiHeadAttention(num_heads=4, key_dim=32, dropout=0.1)
        x_attn = attn(x, x)
        x = Add()([x, x_attn])
        x = LayerNormalization()(x)

    x = GlobalAveragePooling1D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.2)(x)

    out = Dense(horizon)(x)  # ✅ NEW: horizon outputs

    model = Model(inp, out)

    opt = Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    loss = tf.keras.losses.Huber(delta=huber_delta)

    model.compile(optimizer=opt, loss=loss)
    return model


# ======================================================
# 4A. Train model (1-step, backward compatible)
# ======================================================
def train_cnn_bilstm(
    X_train,
    y_train,
    X_val,
    y_val,
    seq_len=24,
    epochs=50,
    batch_size=32,
    use_early_stopping=False,
    patience=8,
    checkpoint_path="artifacts/cnn_bilstm_best.keras",
    save_best_only=True,
    monitor="val_loss",
    learning_rate: float = 3e-4,
    use_attention: bool = True,
    lstm_units: int = 128
):
    Xtr_seq, ytr_seq = build_sequences(X_train, y_train, seq_len)
    Xval_seq, yval_seq = build_sequences(X_val, y_val, seq_len)

    model = build_cnn_bilstm_model(
        seq_len=seq_len,
        n_features=Xtr_seq.shape[2],
        horizon=1,
        learning_rate=learning_rate,
        use_attention=use_attention,
        lstm_units=lstm_units
    )

    callbacks = []

    callbacks.append(ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_best_only=save_best_only,
        mode="min",
        verbose=1
    ))

    callbacks.append(ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=max(2, patience // 2),
        min_lr=1e-6,
        verbose=1
    ))

    if use_early_stopping:
        callbacks.append(EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=False
        ))

    history = model.fit(
        Xtr_seq, ytr_seq,
        validation_data=(Xval_seq, yval_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return model, history


# ======================================================
# 4B. Train model (multi-step, paper-like 48->24)
# ======================================================
def train_cnn_bilstm_multi(
    X_train,
    y_train,
    X_val,
    y_val,
    input_len=48,
    horizon=24,
    epochs=80,
    batch_size=64,
    use_early_stopping=False,
    patience=10,
    checkpoint_path="artifacts/cnn_bilstm_multi_best.keras",
    save_best_only=True,
    monitor="val_loss",
    learning_rate: float = 3e-4,
    use_attention: bool = True,
    lstm_units: int = 128
):
    """
    Multi-step training:
      input_len=48, horizon=24 (paper-like)
    """
    Xtr_seq, ytr_seq = build_sequences_multi(X_train, y_train, input_len=input_len, horizon=horizon)
    Xval_seq, yval_seq = build_sequences_multi(X_val, y_val, input_len=input_len, horizon=horizon)

    model = build_cnn_bilstm_model(
        seq_len=input_len,
        n_features=Xtr_seq.shape[2],
        horizon=horizon,
        learning_rate=learning_rate,
        use_attention=use_attention,
        lstm_units=lstm_units
    )

    callbacks = []

    callbacks.append(ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        save_best_only=save_best_only,
        mode="min",
        verbose=1
    ))

    callbacks.append(ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=max(2, patience // 2),
        min_lr=1e-6,
        verbose=1
    ))

    if use_early_stopping:
        callbacks.append(EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=False
        ))

    history = model.fit(
        Xtr_seq, ytr_seq,
        validation_data=(Xval_seq, yval_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    return model, history


# ======================================================
# 5A. Predict (1-step, backward compatible)
# ======================================================
def predict_cnn_bilstm(model, X_test, seq_len=24):
    Xte_seq = []
    for i in range(seq_len, len(X_test)):
        Xte_seq.append(X_test[i - seq_len:i])
    Xte_seq = np.array(Xte_seq)
    y_pred = model.predict(Xte_seq, verbose=0)
    return y_pred.flatten()


# ======================================================
# 5B. Predict (multi-step)
# ======================================================
def predict_cnn_bilstm_multi(model, X_test, input_len=48, horizon=24):
    """
    Returns:
      y_pred: (n_samples, horizon) where n_samples = len(X_test) - input_len - horizon + 1
    """
    Xte_seq = []
    end = len(X_test) - horizon
    for t in range(input_len, end):
        Xte_seq.append(X_test[t - input_len:t])
    Xte_seq = np.array(Xte_seq)
    y_pred = model.predict(Xte_seq, verbose=0)
    return y_pred  # shape (n_samples, horizon)
