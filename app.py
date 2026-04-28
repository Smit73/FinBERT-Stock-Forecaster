import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import yfinance as yf
from tensorflow.keras.models import load_model
from datetime import timedelta, date
import pipeline 

st.set_page_config(page_title="AI Stock Sniper", layout="wide")
st.title("AI Stock Predictor: Deep Learning + Monte Carlo")

ticker = st.sidebar.text_input("Ticker Symbol", value="NVDA").upper()
model_path = f"{ticker}_model.h5"

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

if not os.path.exists(model_path):
    st.warning(f"⚠️ No AI Brain found for **{ticker}**.")
    if st.button(f"🧠 Train New AI for {ticker}"):
        with st.status("🚀 Launching AI Pipeline...", expanded=True):
            st.write("1️⃣ Scraping Data...")
            success = pipeline.run_full_pipeline(ticker)
            if success:
                st.write("✅ Complete! Reloading...")
                st.rerun()
            else:
                st.error("Failed.")
else:
    st.sidebar.success(f"✅ AI Ready: {ticker}")
    
    try:
        model = load_model(model_path)
        scaler = joblib.load(f"{ticker}_scaler.pkl") 
    except:
        st.error("Scaler mismatch. Please Re-Train.")
        st.stop()
    
    # Get Data
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()
    
    if len(df) < 60:
        st.error("Not enough data.")
    else:
        # 1. LSTM PREDICTION (TOMORROW) 
        # We assume Sentiment is 0.0 (Neutral) for now unless connected to live news
        last_60 = df[['Log_Return', 'SMA_50', 'RSI']].copy()
        last_60['sentiment'] = 0.0 # Placeholder
        # Reorder columns to match training: [Log_Return, sentiment, SMA_50, RSI]
        last_60 = last_60[['Log_Return', 'sentiment', 'SMA_50', 'RSI']].tail(60).values
        
        last_60_scaled = scaler.transform(last_60)
        last_60_seq = last_60_scaled.reshape(1, 60, 4)
        
        pred_scaled = model.predict(last_60_seq, verbose=0)
        
        # Inverse Scale
        dummy_row = np.zeros((1, 4))
        dummy_row[0, 0] = pred_scaled[0][0]
        pred_return = scaler.inverse_transform(dummy_row)[0][0]
        
        current_price = df['Close'].iloc[-1]
        tomorrow_price = current_price * np.exp(pred_return)
        
        # 2. MONTE CARLO SIMULATION (1 YEAR) 
        # This replaces the broken recursive loop.
        # It uses historical stats to generate realistic future paths.
        
        days_forecast = 252 # 1 Trading Year
        num_simulations = 100 # Generate 100 possible futures
        
        # Calculate historical stats from the last year
        # Mean Daily Return (Drift)
        mu = df['Log_Return'].mean()
        # Volatility (Variance)
        sigma = df['Log_Return'].std()
        
        # Generate random paths
        # Formula: Next_Price = Last_Price * e^( (mu - 0.5*sigma^2) + sigma * Random_Z )
        np.random.seed(42) # For consistent results
        simulation_df = pd.DataFrame()
        
        for x in range(num_simulations):
            path = []
            price = current_price
            
            for d in range(days_forecast):
                # Random shock
                drift = mu - (0.5 * sigma**2)
                shock = sigma * np.random.normal() # Z-score
                price = price * np.exp(drift + shock)
                path.append(price)
            
            simulation_df[x] = path
            
        # Calculate the Average Path (The "Target")
        simulation_df['Mean_Path'] = simulation_df.mean(axis=1)
        target_1y = simulation_df['Mean_Path'].iloc[-1]
        
        # DISPLAY
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}")
        col2.metric("AI Prediction (Tomorrow)", f"${tomorrow_price:.2f}", 
                    f"{((tomorrow_price - current_price)/current_price)*100:.2f}%")
        
        delta_1y = ((target_1y - current_price) / current_price) * 100
        col3.metric("Monte Carlo 1-Year Target", f"${target_1y:.2f}", f"{delta_1y:.2f}%")
        
        # CHARTING 
        st.subheader(f"1-Year Projection: {ticker}")
        
        last_date = df.index[-1].date()
        future_dates = [last_date + timedelta(days=x) for x in range(1, days_forecast + 1)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 1. Plot History
        ax.plot(df.index[-252:], df['Close'].tail(252), label='History', color='black', linewidth=2)
        
        # 2. Plot Simulation Paths (Faint Lines)
        for x in range(min(50, num_simulations)): # Show top 50 paths
            ax.plot(future_dates, simulation_df[x], color='green', alpha=0.05)
            
        # 3. Plot Mean Path (Target Line)
        ax.plot(future_dates, simulation_df['Mean_Path'], label='Projected Average', color='blue', linestyle='--', linewidth=2)
        
        ax.set_title(f"Monte Carlo Simulation ({num_simulations} scenarios)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        with st.expander("ℹ️ Why Monte Carlo?"):
            st.write("""
            LSTMs are excellent for short-term pattern recognition (Tomorrow), but they become unstable when predicting 
            far into the future (1 Year) because errors compound.
            
            **Monte Carlo Simulation** is the industry standard for long-term forecasting. 
            It uses the stock's historical Volatility and Drift to generate a 'Cone of Probability', 
            showing the range of likely outcomes rather than a single guess.
            """)