# NLP-Enhanced Stock Strategy Analysis using Financial News Sentiment and Historical Backtesting

A comprehensive market analysis and historical comparison dashboard built with Python and Streamlit. This application extends traditional stock strategy backtesting by integrating **Financial News Collection**, **Classical NLP Preprocessing**, **FinBERT Sentiment Analysis**, **BART/T5 News Summarization**, and **Signal Agreement / Conflict Detection**.

---

## 🌟 Key Features

* **Historical Stock Backtesting:**
  * Interactive Yahoo Finance stock data downloading (`yfinance`).
  * 20-Day and 50-Day Moving Average Crossover strategy evaluation vs Buy & Hold benchmark.
  * Calculates Strategy Return, Buy & Hold Return, Outperformance %, Total Signals, and Portfolio Equity Curves.
* **Current Technical Analysis:**
  * Real-time technical trend evaluation comparing latest Close Price, MA20, and MA50.
* **Financial News Collection & Demo Mode:**
  * Collects latest financial news via REST APIs (Finnhub, NewsAPI, or GNews).
  * Seamless **DEMO MODE** fallback when API key is missing or offline so presentations always run smoothly.
* **Classical NLP Preprocessing Pipeline:**
  * Step-by-step demonstration of Text Cleaning → Tokenization → Stopword Handling → Lemmatization (`nltk`).
  * Maintains dual text representations (Representation A for classic NLP demo, Representation B for natural language transformer models).
* **FinBERT Domain Sentiment Analysis:**
  * Uses pretrained `ProsusAI/finbert` to classify financial text into Positive, Neutral, and Negative categories.
  * Exposes exact percentage probabilities for every article and computes an Aggregate Sentiment Score.
* **BART/T5 News Summarization:**
  * Generates executive financial summaries synthesizing top news stories using neural summarization.
* **Signal Agreement / Conflict Detection:**
  * Evaluates alignment between Historical Performance, Current Technical Condition, and Current News Sentiment.
  * Provides an explainable **Market Context Score (0 to 100)** and context badge (🟢 Positive Context, 🟡 Mixed Context, 🔴 Negative Context).
* **Comprehensive Export Options:**
  * Download historical price data CSV, performance summary TXT, news sentiment analysis CSV, and full NLP market analysis reports.

---

## 🏗 System Architecture

```
HISTORICAL ANALYSIS                  CURRENT MARKET ANALYSIS                 CURRENT NEWS NLP
   Stock Data                           Stock Data                            Financial News
       ↓                                    ↓                                       ↓
Moving Average Strategy                 Current Price                           Text Cleaning
       ↓                                 MA20 & MA50                                ↓
Historical Backtest               Current Technical Condition                   Normalization
       ↓                                    ↓                                       ↓
Strategy Return & Equity             Current Market Context                     Tokenization
       ↓                                                                            ↓
Historical Trend Context                                                        Stopword Removal
                                                                                    ↓
                                                                               Lemmatization
                                                                                    ↓
                                                                               FinBERT Model
                                                                                    ↓
                                                                          Pos / Neu / Neg Probs
                                                                                    ↓
                                                                         Aggregate Sentiment Score
                                                                                    ↓
                                                                          BART/T5 Summarization
                                                                                    ↓
                                                                             News Summary
                                                  ↓
                                        INTEGRATED MARKET CONTEXT
                                                  ↓
                                     Market Context Score & Badge
                                                  ↓
                                  Signal Agreement / Conflict Analysis
```

---

## 📦 Setup & Installation

### 1. Clone & Prerequisites

Ensure Python 3.10+ is installed on your system.

```bash
cd Stock-Market-Strategy-Backtester
```

### 2. Install Dependencies

Install required libraries from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. API Key Configuration (Optional)

The application includes a built-in **DEMO MODE** that works out of the box without any API keys. 

To enable live news collection from Finnhub or NewsAPI:
1. Create a `.streamlit` directory inside the project root if it doesn't exist.
2. Create a `secrets.toml` file inside `.streamlit/`:

```toml
# .streamlit/secrets.toml
FINNHUB_API_KEY = "your_finnhub_api_key_here"
# NEWS_API_KEY = "your_newsapi_key_here"
```

---

## 🚀 Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

---

## 📄 License & Disclaimer

**Educational & Analytical Purpose Only:** 
This application is strictly for academic, research, and educational purposes. Historical strategy performance and financial news sentiment do NOT guarantee future market results. This application does NOT provide personalized financial or investment advice.
