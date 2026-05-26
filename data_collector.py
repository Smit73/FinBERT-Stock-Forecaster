import finnhub
import pandas as pd
import time
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import requests

# CONFIG
#  KEY
try:
    API_KEY = st.secrets["FINNHUB_API_KEY"]
except Exception:
    API_KEY = None # If the safe is missing or empty, set to None

def safe_convert_date(ts):
    """
    Safely converts a timestamp to a date object.
    Handles seconds (10 digits) and milliseconds (13 digits).
    """
    try:
        ts = int(ts)
        # If timestamp is massive (13 digits), it's in milliseconds -> convert to seconds
        if ts > 10000000000: 
            ts = ts / 1000
        return datetime.fromtimestamp(ts).date()
    except (ValueError, OSError, OverflowError):
        # If conversion fails (e.g. negative time on Windows), return Today
        return datetime.now().date()

def fetch_data_for_symbol(ticker):
    print(f"\n Starting Data Collection for {ticker}...")
    
    # 1. Calculate Date Range (1 Year)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    
    print(f"   Timeframe: {start_date} to {end_date}")

    # 2. Get Price Data (Stealthy way)
    print("   📉 Downloading Price Data...")
    try:
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
        })
        prices = yf.Ticker(ticker, session=session).history(start=start_date, end=end_date)
        
        if prices.empty:
            print("      ❌ No price data found. Yahoo might be blocking the request.")
            return False
        prices.to_csv(f"{ticker}_prices.csv")
    except Exception as e:
        print(f"      ❌ yfinance Error: {e}")
        return False
    
    # 3. Get News Data
    print("   📰 Downloading News Data (this takes ~30s)...")
    if not API_KEY:
        print("      ❌ STOP: Finnhub API Key is missing!")
        return False

    try:
        finnhub_client = finnhub.Client(api_key=API_KEY)
        all_articles = []
        
        current_date = start_date
        while current_date < end_date:
            _from = current_date.strftime("%Y-%m-%d")
            # Fetch 2 weeks at a time
            next_date = current_date + timedelta(days=14)
            if next_date > end_date: next_date = end_date
            _to = next_date.strftime("%Y-%m-%d")
            
            try:
                news = finnhub_client.company_news(ticker, _from=_from, to=_to)
                
                if news:
                    for item in news:
                        # Skip items with missing timestamps
                        if 'datetime' not in item:
                            continue
                            
                        # Use Safe Converter
                        article_date = safe_convert_date(item['datetime'])
                        
                        all_articles.append({
                            'date': article_date,
                            'title': item.get('headline', ''),
                            'desc': item.get('summary', '')
                        })
            except Exception as api_error:
                print(f"      ⚠️ API chunk error (skipping chunk): {api_error}")
            
            time.sleep(0.2) # Rate limit safety
            current_date = next_date + timedelta(days=1)
            
        news_df = pd.DataFrame(all_articles)
        
        if not news_df.empty:
            news_df = news_df.drop_duplicates(subset=['title'])
            news_df.to_csv(f"{ticker}_news.csv", index=False)
            print(f"   ✅ Found {len(news_df)} news articles.")
        else:
            print("   ⚠️ No news found (creating empty file to prevent crash).")
            # Create dummy file so pipeline continues
            pd.DataFrame(columns=['date', 'title', 'desc']).to_csv(f"{ticker}_news.csv", index=False)
            
        return True
        
    except Exception as e:
        print(f"   ❌ Critical Finnhub Error: {e}")
        return False

if __name__ == "__main__":
    fetch_data_for_symbol("NVDA")