import pandas as pd
import numpy as np

# CONCEPT: Vectorized Math 
# We write functions that work on entire arrays, not single numbers.
def calculate_rsi(data, window=14):
    """
    Calculates the Relative Strength Index (RSI).
    RSI tells us if a stock is 'overbought' (>70) or 'oversold' (<30).
    """
    delta = data.diff() # 'diff' is Today's Price - Yesterday's Price
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    # Calculate smooth averages (Exponential Moving Average)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi
def prepare_data(ticker):
    print(f" Processing data for {ticker}...")
    
    # 1. Load the CSVs
    # 'parse_dates' tells pandas to treat the Date column as actual Time objects, not Strings.
    try:
        prices_df = pd.read_csv(f"{ticker}_prices.csv", parse_dates=['Date'])
        # FinBERT data might use 'date' (lowercase), so we standardize
        sentiment_df = pd.read_csv(f"{ticker}_daily_sentiment.csv", parse_dates=['date'])
        # --- THE FIX ---
    # Convert both date columns to datetime objects first
        prices_df['Date'] = pd.to_datetime(prices_df['Date'], utc=True)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'], utc=True)

    # Normalize them to just the DATE (removes hours, minutes, timezones)
        prices_df['Date'] = prices_df['Date'].dt.date
        sentiment_df['date'] = sentiment_df['date'].dt.date        
        sentiment_df.rename(columns={'date': 'Date'}, inplace=True) # Rename to match
    except FileNotFoundError:
        print(" Files not found! Run Step 1 and 2 first.")
        return

    # 2. Merge (The SQL 'Left Join')
    # We keep ALL price rows. If we have sentiment for that day, we attach it.
    # If no sentiment exists for that day, we get NaN (empty).
    merged_df = pd.merge(prices_df, sentiment_df, on='Date', how='left')
    
    # 3. Handle Missing Sentiment
    # If there was no news, the sentiment is 0 (Neutral), not Null.
    merged_df['sentiment'] = merged_df['sentiment'].fillna(0)
    
    # 4. Feature Engineering (The Math)
    print("    Calculating Technical Indicators...")
    
    # SMA_50: Average of the last 50 days
    merged_df['SMA_50'] = merged_df['Close'].rolling(window=50).mean()
    
    # RSI: The momentum
    merged_df['RSI'] = calculate_rsi(merged_df['Close'])
    
    # 5. The "Target" (What we want to predict)
    # We want to predict TOMORROW'S Close price using TODAY'S data.
    # So we shift the Close column UP by 1 row.
    # Today's row will now contain Tomorrow's price in the 'Target' column.
    merged_df['Target'] = merged_df['Close'].shift(-1)
    
    # 6. Cleanup
    # Calculating SMA_50 creates 49 empty rows at the start (no previous data).
    # Shift(-1) creates 1 empty row at the very end (no tomorrow yet).
    # We drop these N/A rows so the AI doesn't crash.
    final_df = merged_df.dropna()
    
    print(f"    Final Dataset: {len(final_df)} rows ready for AI.")
    print(final_df[['Date', 'Close', 'sentiment', 'SMA_50', 'Target']].tail())
    
    # Save
    final_df.to_csv(f"{ticker}_final_data.csv", index=False)
    print("💾 Saved to final_data.csv")

if __name__ == "__main__":
    prepare_data("NVDA")