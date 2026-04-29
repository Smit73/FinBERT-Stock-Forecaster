import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

def create_sequences(dataset, lookback):
    X, y = [], []
    for i in range(lookback, len(dataset)):
        X.append(dataset[i-lookback:i])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

def train_model(ticker):
    print(f"🧠 Training LSTM (Standardized) for {ticker}...")
    
    file_path = f"{ticker}_final_data.csv"
    if not os.path.exists(file_path):
        print(f"❌ {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    # 1. Log Returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()
    
    feature_cols = ['Log_Return', 'sentiment', 'SMA_50_Ratio', 'RSI']
    
    # 2. SCALING FIX: Use StandardScaler
    # This centers the data around 0. Mean = 0, Std Dev = 1.
    scaler = StandardScaler() 
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    # 3. Create Sequences
    LOOKBACK = 60
    X, y = create_sequences(scaled_data, LOOKBACK)
    
    # Train/Test Split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # 4. Build Model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=16, epochs=15, verbose=1)
    
    # Save
    model.save(f"{ticker}_model.h5")
    joblib.dump(scaler, f"{ticker}_scaler.pkl")
    print(f"✅ Model saved: {ticker}_model.h5")

if __name__ == "__main__":
    train_model("NVDA")