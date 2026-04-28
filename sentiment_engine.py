import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

 #Load the AI model FinBERT
print("Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def estimate_sentiment(news_list):
    """
    Input: A list of news headlines (text).
    Output: A DataFrame with the original text and its sentiment score.
    """
    if not news_list: return []
    
    # CONFIG
    # Process 50 articles at a time. 
    # If you still crash, lower this to 20.
    BATCH_SIZE = 50 
    
    total_articles = len(news_list)
    print(f"   ...Processing {total_articles} articles in batches of {BATCH_SIZE}...")
    
    all_scores = []
    
    # --- THE BATCH LOOP --- Analyzes news in small batches to prevent crashing RAM
    for i in range(0, total_articles, BATCH_SIZE):
        # 1. Slice the batch
        batch_text = news_list[i : i + BATCH_SIZE]

    # Pre-processing (Tokenization)
    # PERSONAL NOTE: The AI doesn't read words, it reads numbers (tokens).
    # PERSONAL NOTE: return_tensors='pt' means returns PyTorch Tensors instead of standard Python lists. The model will crash if you feed it lists.
    # PERSONAL NOTE: padding=True means "If sentences are different lengths, pad them with zeros so they match."
        inputs = tokenizer(batch_text, return_tensors="pt", padding=True, truncation=True) # We use truncation=True because FinBERT can only handle 512 "tokens"

    #The prediction
    #pass the inputs into the model to get raw scores
        with torch.no_grad(): #Disables gradient tracking (Calculus) to save RAM and speed up prediction, since we aren't training the model onlt predicting.
            outputs = model(**inputs)# Python automatically converts this to: model(input_ids=..., attention_mask=...) ::: Passes the tokenized data (IDs + Masks) to the model cleanly without extracting fields manually.

    #convert to probabilities with softmax:which turns raw numbers in percentages (0.0-1.0)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # The model returns 3 numbers per headline: [Positive, Negative, Neutral]
    # We want a single "Compound Score" from -1 (Bad) to +1 (Good).
    # Logic: Score = Positive_Probability - Negative_Probability
    
        positive_probs = predictions[:, 0].tolist() # Column 0 is Positive
        negative_probs = predictions[:, 1].tolist() # Column 1 is Negative
    # (Column 2 is Neutral, we ignore it for the score calculation)

    # This is a "Pythonic" one-line for-loop.
    # It says: "Calculate p - n for every p and n in our lists."
        sentiment_scores = [p - n for p, n in zip(positive_probs, negative_probs)]

    # 6. Append to master list
        all_scores.extend(sentiment_scores)

    # Optional- Print progress every 500 items
        if (i + BATCH_SIZE) % 500 == 0:
            print(f"      Analyzed {min(i + BATCH_SIZE, total_articles)}/{total_articles}...")

    return all_scores

def run_analysis(ticker):
    """
    Main logic wrapper. Reads {ticker}_news.csv -> Writes {ticker}_daily_sentiment.csv
    """
    print(f"🧠 Running Sentiment Analysis for {ticker}...")
    
    input_file = f"{ticker}_news.csv"
    if not os.path.exists(input_file):
        print(f"❌ {input_file} not found.")
        return

    df = pd.read_csv(input_file)
    
    if df.empty:
        print("⚠️ No news to analyze.")
        # Create an empty sentiment file so data_processor doesn't crash
        pd.DataFrame(columns=['date', 'sentiment']).to_csv(f"{ticker}_daily_sentiment.csv", index=False)
        return

    # Combine Title + Desc
    df['title'] = df['title'].fillna('')
    df['desc'] = df['desc'].fillna('')
    df['full_text'] = df['title'] + ". " + df['desc']
    
    # Run AI
    scores = estimate_sentiment(df['full_text'].tolist())
    df['sentiment'] = scores
    
    # Aggregate by Date
    daily_scores = df.groupby('date')['sentiment'].mean().reset_index()
    daily_scores.to_csv(f"{ticker}_daily_sentiment.csv", index=False)
    print(f"✅ Sentiment saved to {ticker}_daily_sentiment.csv")

# - KEEPS IT TESTABLE 
if __name__ == "__main__":
    #  can still run this file manually to test it
    run_analysis("NVDA")

