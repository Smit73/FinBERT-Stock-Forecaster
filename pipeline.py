import data_collector
import sentiment_engine
import data_processor
import model_trainer
import os

def run_full_pipeline(ticker):
    """
    Runs the entire AI workflow for a specific stock symbol.
    Returns True if successful, False if failed.
    """
    ticker = ticker.upper()
    print(f"//////////////////////////////////////////////////")
    print(f"(pipeline.py) STARTING AI PIPELINE FOR: {ticker}")
    print(f"//////////////////////////////////////////////////")
    
    # first Collect Data
    success = data_collector.fetch_data_for_symbol(ticker)
    if not success: return False
    
    # second Analyze Sentiment
    print("(pipeline.py) Running Sentiment Analysis...")
    sentiment_engine.run_analysis(ticker) 
    
    # third Process & Merge
    print("(pipeline.py) Processing Data...")
    data_processor.prepare_data(ticker)
    
    # lastly Train Model
    print("(pipeline.py) Training LSTM Model...")
    model_trainer.train_model(ticker) 
    
    print(f"(pipeline.py) PIPELINE COMPLETE. Model saved as {ticker}_model.h5")
    return True