import os
import pandas as pd
import yfinance as yf

# Company name mapping for popular Indian and global tickers
TICKER_NAME_MAP = {
    "TCS.NS": "Tata Consultancy Services (TCS)",
    "RELIANCE.NS": "Reliance Industries Limited",
    "INFY.NS": "Infosys Limited",
    "HDFCBANK.NS": "HDFC Bank Limited",
    "ICICIBANK.NS": "ICICI Bank Limited",
    "TATAMOTORS.NS": "Tata Motors Limited",
    "SBIN.NS": "State Bank of India",
    "BHARTIARTL.NS": "Bharti Airtel Limited",
    "WIPRO.NS": "Wipro Limited",
    "ITC.NS": "ITC Limited",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc. (Google)",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corporation",
    "META": "Meta Platforms Inc."
}


def get_company_name(ticker: str) -> str:
    """Returns readable company name for ticker or formatted ticker if unknown."""
    if not ticker:
        return ""
    ticker_clean = ticker.strip().upper()
    if ticker_clean in TICKER_NAME_MAP:
        return TICKER_NAME_MAP[ticker_clean]
    
    # Clean symbol fallback
    base_symbol = ticker_clean.split('.')[0]
    return f"{base_symbol} ({ticker_clean})"


def download_stock_data(ticker: str, start_date: str, end_date: str):
    """
    Downloads historical stock data from Yahoo Finance.
    Returns None if ticker is invalid or data is unavailable.
    """

    print("\nDownloading historical stock data...\n")

    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # Fix MultiIndex columns returned by latest yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

    try:
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column])
    except Exception:
        return None

    os.makedirs("../data", exist_ok=True)
    try:
        df.to_csv("../data/historical_stock_data.csv", index=False)
    except Exception:
        pass

    print(f"Downloaded {len(df)} rows successfully.")

    return df