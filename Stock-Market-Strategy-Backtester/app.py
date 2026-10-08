import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.append(str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from backtester import backtest_strategy
from data_loader import download_stock_data, get_company_name
from indicators import calculate_indicators
from strategy import generate_signals
from news_fetcher import fetch_financial_news
from nlp_preprocessor import preprocess_article
from sentiment_analyzer import analyze_news_sentiment
from summarizer import generate_news_summary
from market_analysis import (
    analyze_current_technical_condition,
    compute_market_context_score,
)
from visualization import create_sentiment_chart


st.set_page_config(
    page_title="NLP-Enhanced Stock Strategy Analysis",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Market Strategy Backtester & NLP Market Analysis")

st.markdown(
    "Evaluate a **Moving Average Crossover** trading strategy using historical stock data "
    "from Yahoo Finance, combined with **NLP Financial News Sentiment Analysis & BART/T5 Summarization**."
)

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.header("Strategy Inputs")

ticker = st.sidebar.text_input("Ticker", value="TCS.NS")

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2020-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    value=date.today() - timedelta(days=1)
)

capital = st.sidebar.number_input(
    "Initial Capital",
    value=100000,
    step=10000
)

run = st.sidebar.button("🚀 Run Backtest & Analysis", use_container_width=True)

st.sidebar.info("Examples:\n\n• TCS.NS\n\n• RELIANCE.NS\n\n• INFY.NS\n\n• AAPL\n\n• MSFT")

# -----------------------------
# Session State
# -----------------------------

if "df" not in st.session_state:
    st.session_state.df = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "error" not in st.session_state:
    st.session_state.error = None

if "news_df" not in st.session_state:
    st.session_state.news_df = None

if "sentiment_metrics" not in st.session_state:
    st.session_state.sentiment_metrics = None

if "news_status" not in st.session_state:
    st.session_state.news_status = ""

if "news_summary" not in st.session_state:
    st.session_state.news_summary = ""

# -----------------------------
# Run Backtest & Analysis
# -----------------------------

if run:
    # Reset previous state
    st.session_state.df = None
    st.session_state.summary = None
    st.session_state.error = None
    st.session_state.news_df = None
    st.session_state.sentiment_metrics = None
    st.session_state.news_status = ""
    st.session_state.news_summary = ""

    error_msg = None

    with st.spinner("Downloading stock data, running backtest, and analyzing news NLP..."):

        df = download_stock_data(ticker, str(start_date), str(end_date))

        if df is None:
            error_msg = "❌ Invalid ticker or no data found. Please enter a valid Yahoo Finance ticker."
        else:
            try:
                df = calculate_indicators(df)
                df = generate_signals(df)
                df = backtest_strategy(df, capital)

                strategy_return = (
                    (df["Portfolio Value"].iloc[-1] - capital) / capital
                ) * 100

                buy_hold_return = (
                    (df["Buy Hold Value"].iloc[-1] - capital) / capital
                ) * 100

                outperformance = strategy_return - buy_hold_return
                total_signals = df["Signal"].diff().fillna(0).ne(0).sum()

                summary = {
                    "strategy_return": strategy_return,
                    "buy_hold_return": buy_hold_return,
                    "outperformance": outperformance,
                    "signals": total_signals,
                    "final_portfolio": df["Portfolio Value"].dropna().iloc[-1],
                    "latest_price": df["Close"].dropna().iloc[-1],
                    "highest": df["High"].max(),
                    "lowest": df["Low"].min(),
                }

                st.session_state.df = df
                st.session_state.summary = summary

                # Fetch and analyze financial news NLP
                raw_news_df, status_msg = fetch_financial_news(ticker)
                st.session_state.news_status = status_msg

                if raw_news_df is not None and not raw_news_df.empty:
                    processed_news_df, sent_metrics = analyze_news_sentiment(raw_news_df)
                    st.session_state.news_df = processed_news_df
                    st.session_state.sentiment_metrics = sent_metrics
                    
                    summary_text, _ = generate_news_summary(processed_news_df)
                    st.session_state.news_summary = summary_text

            except ValueError as e:
                error_msg = f"❌ {e}"

            except Exception as e:
                error_msg = f"❌ Something went wrong while processing the stock data or analysis: {e}"

    # Show error or success AFTER spinner closes
    if error_msg:
        st.session_state.error = error_msg
    else:
        st.success("✅ Backtest and NLP Analysis completed successfully!")

# Show persistent error if any
if st.session_state.error:
    st.error(st.session_state.error)
    st.stop()

# -----------------------------
# Nothing executed yet
# -----------------------------

if st.session_state.df is None:
    st.info("Configure inputs in the sidebar and click **Run Backtest & Analysis** to get started.")
    st.stop()

df = st.session_state.df
summary = st.session_state.summary
news_df = st.session_state.news_df
sentiment_metrics = st.session_state.sentiment_metrics
news_status = st.session_state.news_status
news_summary = st.session_state.news_summary
company_name = get_company_name(ticker)

# -----------------------------
# Top Metrics
# -----------------------------

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("📈 Strategy Return", f"{summary['strategy_return']:.2f}%")
m2.metric("📊 Buy & Hold", f"{summary['buy_hold_return']:.2f}%")
m3.metric("⚖️ Outperformance", f"{summary['outperformance']:.2f}%")
m4.metric("🔄 Signals", int(summary["signals"]))
m5.metric("📅 Trading Days", len(df))

st.divider()

m6, m7, m8, m9, m10 = st.columns(5)
m6.metric("💵 Initial Capital", f"₹{capital:,.0f}")
m7.metric("📈 Latest Price", f"₹{summary['latest_price']:.2f}")
m8.metric("💼 Portfolio Value", f"₹{summary['final_portfolio']:,.0f}")
m9.metric("📈 Highest Price", f"₹{summary['highest']:.2f}")
m10.metric("📉 Lowest Price", f"₹{summary['lowest']:.2f}")

st.divider()

overview_tab, charts_tab, ai_analysis_tab, data_tab, download_tab, about_tab = st.tabs(
    ["📊 Overview", "📈 Charts", "🧠 AI Market Analysis", "📄 Dataset", "⬇ Downloads", "ℹ About"]
)

# ======================================================
# OVERVIEW TAB
# ======================================================

with overview_tab:

    st.subheader("Performance Summary")

    summary_df = pd.DataFrame({
        "Metric": [
            "Ticker & Asset",
            "Initial Capital",
            "Final Portfolio Value",
            "Strategy Return",
            "Buy & Hold Return",
            "Outperformance",
            "Trading Signals",
            "Trading Days"
        ],
        "Value": [
            f"{ticker} ({company_name})",
            f"₹{capital:,.2f}",
            f"₹{summary['final_portfolio']:,.2f}",
            f"{summary['strategy_return']:.2f}%",
            f"{summary['buy_hold_return']:.2f}%",
            f"{summary['outperformance']:.2f}%",
            int(summary["signals"]),
            len(df)
        ]
    })

    st.dataframe(summary_df, hide_index=True, use_container_width=True)

# ======================================================
# CHARTS TAB
# ======================================================

with charts_tab:

    row1_col1, row1_col2 = st.columns(2)

    with row1_col1:
        st.subheader("📈 Stock Price")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["Date"], df["Close"], linewidth=2)
        ax.set_title("Closing Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    with row1_col2:
        st.subheader("📊 Moving Average")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["Date"], df["Close"], label="Close")
        ax.plot(df["Date"], df["MA20"], label="20 MA")
        ax.plot(df["Date"], df["MA50"], label="50 MA")
        ax.set_title("Moving Average Crossover")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.subheader("🟢 Buy / Sell Signals")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["Date"], df["Close"], label="Close")
        buy = df[df["Signal"] == 1]
        sell = df[df["Signal"] == -1]
        ax.scatter(buy["Date"], buy["Close"], marker="^", color="green", s=80, label="BUY")
        ax.scatter(sell["Date"], sell["Close"], marker="v", color="red", s=80, label="SELL")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

    with row2_col2:
        st.subheader("💰 Portfolio Comparison")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["Date"], df["Portfolio Value"], label="Strategy")
        ax.plot(df["Date"], df["Buy Hold Value"], label="Buy & Hold")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)

# ======================================================
# 🧠 AI MARKET ANALYSIS TAB (NEW MAJOR TAB)
# ======================================================

with ai_analysis_tab:
    st.subheader(f"🧠 AI Market Analysis: {ticker} ({company_name})")
    
    if news_status:
        st.info(news_status)
        
    tech_info = analyze_current_technical_condition(df)
    
    if sentiment_metrics is None:
        sentiment_metrics = {
            "avg_positive": 0.33,
            "avg_neutral": 0.34,
            "avg_negative": 0.33,
            "aggregate_score": 0.0,
            "qualitative_label": "Neutral / Unavailable",
            "pos_count": 0,
            "neu_count": 0,
            "neg_count": 0,
            "total_articles": 0,
            "is_finbert_used": False
        }

    market_context = compute_market_context_score(summary, tech_info, sentiment_metrics)
    
    # --------------------------------------------------
    # 1. Current Market Snapshot
    # --------------------------------------------------
    st.markdown("### 📊 Current Market Snapshot")
    snap1, snap2, snap3, snap4 = st.columns(4)
    snap1.metric("Current Price", f"₹{tech_info['current_price']:,.2f}")
    snap2.metric("20-Day MA (MA20)", f"₹{tech_info['ma20']:,.2f}")
    snap3.metric("50-Day MA (MA50)", f"₹{tech_info['ma50']:,.2f}")
    snap4.metric("Technical Condition", tech_info['condition_label'])
    
    st.divider()

    # --------------------------------------------------
    # 2. News Sentiment Overview
    # --------------------------------------------------
    st.markdown("### 📈 Current News Sentiment Overview")
    
    s_col1, s_col2 = st.columns([1, 1])
    
    with s_col1:
        st.write(f"**Articles Analyzed:** {sentiment_metrics['total_articles']}")
        st.write(f"**Overall News Sentiment:** `{sentiment_metrics['qualitative_label']}`")
        st.write(f"**Sentiment Score:** `{sentiment_metrics['aggregate_score']:+.2f}` (Range: -1.0 to +1.0)")
        
        pos_pct = sentiment_metrics['avg_positive'] * 100
        neu_pct = sentiment_metrics['avg_neutral'] * 100
        neg_pct = sentiment_metrics['avg_negative'] * 100
        
        st.markdown(f"🟢 **Positive:** {pos_pct:.1f}% ({sentiment_metrics['pos_count']} articles)")
        st.progress(min(1.0, max(0.0, pos_pct / 100)))
        
        st.markdown(f"🟡 **Neutral:** {neu_pct:.1f}% ({sentiment_metrics['neu_count']} articles)")
        st.progress(min(1.0, max(0.0, neu_pct / 100)))
        
        st.markdown(f"🔴 **Negative:** {neg_pct:.1f}% ({sentiment_metrics['neg_count']} articles)")
        st.progress(min(1.0, max(0.0, neg_pct / 100)))

    with s_col2:
        chart_fig = create_sentiment_chart(sentiment_metrics)
        st.pyplot(chart_fig)
        plt.close(chart_fig)

    st.divider()

    # --------------------------------------------------
    # 3. Recent News Feed & Per-Article FinBERT Analysis
    # --------------------------------------------------
    st.markdown("### 📰 Recent Financial News & FinBERT Classification")
    
    if news_df is not None and not news_df.empty:
        display_news_df = news_df.copy()
        display_news_df["Pos %"] = (display_news_df["positive_probability"] * 100).map("{:.1f}%".format)
        display_news_df["Neu %"] = (display_news_df["neutral_probability"] * 100).map("{:.1f}%".format)
        display_news_df["Neg %"] = (display_news_df["negative_probability"] * 100).map("{:.1f}%".format)
        
        table_cols = ["headline", "source", "publication_date", "Pos %", "Neu %", "Neg %", "sentiment"]
        st.dataframe(
            display_news_df[table_cols].rename(columns={
                "headline": "Headline",
                "source": "Source",
                "publication_date": "Date",
                "sentiment": "FinBERT Sentiment"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No financial news articles retrieved.")

    st.divider()

    # --------------------------------------------------
    # 4. Show NLP Processing Demonstration (Academic Feature)
    # --------------------------------------------------
    with st.expander("🔬 Show NLP Processing Demonstration (Step-by-Step Pipeline)"):
        st.markdown("""
        This section demonstrates classical NLP preprocessing (Cleaning → Normalization → Tokenization → Stopwords Removal → Lemmatization)
        alongside transformer-based FinBERT natural language context processing.
        """)
        
        if news_df is not None and not news_df.empty:
            selected_headline = st.selectbox(
                "Select an article to inspect NLP preprocessing steps:",
                options=news_df["headline"].tolist()
            )
            selected_row = news_df[news_df["headline"] == selected_headline].iloc[0]
            
            sample_headline = selected_row["headline"]
            sample_desc = selected_row.get("description", "")
            sample_pos = selected_row["positive_probability"] * 100
            sample_neu = selected_row["neutral_probability"] * 100
            sample_neg = selected_row["negative_probability"] * 100
            sample_label = selected_row["sentiment"]
            
            nlp_res = preprocess_article(sample_headline, sample_desc)
            
            st.markdown("#### Step 1: Raw Text Input")
            st.code(nlp_res["raw_text"])
            
            st.markdown("#### Step 2: Cleaned & Normalized Text")
            st.code(nlp_res["cleaned_text"])
            
            st.markdown("#### Step 3: Tokenization")
            st.write(nlp_res["raw_tokens"])
            
            st.markdown("#### Step 4: Stopword Removal")
            st.write(nlp_res["tokens_no_stopwords"])
            
            st.markdown("#### Step 5: Lemmatization (Representation A)")
            st.code(nlp_res["representation_a"])
            
            st.markdown("#### Step 6: FinBERT Sentiment Probabilities (Representation B Input)")
            st.write(f"**Input to FinBERT:** `{nlp_res['representation_b']}`")
            st.write(f"🟢 **Positive:** {sample_pos:.1f}% | 🟡 **Neutral:** {sample_neu:.1f}% | 🔴 **Negative:** {sample_neg:.1f}%")
            st.write(f"**Predicted Class:** `{sample_label}`")

    st.divider()

    # --------------------------------------------------
    # 5. Financial News Summarization
    # --------------------------------------------------
    st.markdown("### 📝 Recent Financial News Summary")
    if news_summary:
        st.success(news_summary)
    else:
        st.write("News summary unavailable.")

    st.divider()

    # --------------------------------------------------
    # 6. Signal Comparison & Conflict Detection
    # --------------------------------------------------
    st.markdown("### 🔎 Signal Comparison & Conflict Detection")
    st.dataframe(market_context["comparison_table"], use_container_width=True, hide_index=True)

    st.divider()

    # --------------------------------------------------
    # 7. Overall Market Context
    # --------------------------------------------------
    st.markdown("### 🧭 Overall Market Context")
    
    ctx_col1, ctx_col2 = st.columns([2, 1])
    
    with ctx_col1:
        st.subheader(f"{market_context['context_badge']}")
        st.markdown(f"**Status Category:** `{market_context['status_category']}`")
        st.write(market_context["explanation"])
        
    with ctx_col2:
        st.metric(
            "Market Context Score",
            f"{market_context['market_context_score']} / 100"
        )
        
    st.warning(f"⚠️ **Disclaimer:** {market_context['disclaimer']}")

# ======================================================
# DATA TAB
# ======================================================

with data_tab:
    st.subheader("Historical Dataset")
    st.write(f"Rows : {len(df)} | Columns : {len(df.columns)}")
    st.dataframe(df, use_container_width=True)

# ======================================================
# DOWNLOAD TAB
# ======================================================

with download_tab:
    st.subheader("Download Results")
    st.write("Download historical datasets, performance reports, and news sentiment analyses.")

    d_col1, d_col2 = st.columns(2)
    
    with d_col1:
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Historical Data CSV",
            csv,
            file_name=f"{ticker}_historical_stock_data.csv",
            mime="text/csv",
            use_container_width=True
        )

        report = f"""
====================================
Performance Summary Report
====================================

Ticker             : {ticker} ({company_name})
Initial Capital    : ₹{capital:,.2f}
Final Portfolio    : ₹{summary['final_portfolio']:,.2f}
Strategy Return    : {summary['strategy_return']:.2f} %
Buy & Hold Return  : {summary['buy_hold_return']:.2f} %
Outperformance     : {summary['outperformance']:.2f} %
Signals Generated  : {int(summary['signals'])}
Trading Days       : {len(df)}
"""
        st.download_button(
            "📄 Download Performance Summary TXT",
            report,
            file_name=f"{ticker}_performance_summary.txt",
            mime="text/plain",
            use_container_width=True
        )

    with d_col2:
        if news_df is not None and not news_df.empty:
            news_csv = news_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download News Sentiment CSV",
                news_csv,
                file_name=f"{ticker}_news_sentiment_analysis.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            full_nlp_report = f"""
======================================================
NLP-ENHANCED MARKET ANALYSIS REPORT
======================================================

Date & Time of Analysis : {date.today()}
Ticker                  : {ticker} ({company_name})

1. HISTORICAL PERFORMANCE SUMMARY
------------------------------------------------------
Initial Capital         : ₹{capital:,.2f}
Final Portfolio Value   : ₹{summary['final_portfolio']:,.2f}
Strategy Return         : {summary['strategy_return']:.2f} %
Buy & Hold Return       : {summary['buy_hold_return']:.2f} %
Outperformance          : {summary['outperformance']:.2f} %
Total Signals           : {int(summary['signals'])}

2. CURRENT TECHNICAL CONDITION
------------------------------------------------------
Latest Price            : ₹{summary['latest_price']:.2f}
20-Day Moving Average   : ₹{df['MA20'].iloc[-1]:.2f}
50-Day Moving Average   : ₹{df['MA50'].iloc[-1]:.2f}
Technical Trend         : {tech_info['condition_label']}

3. FINANCIAL NEWS SENTIMENT ANALYSIS
------------------------------------------------------
Articles Analyzed       : {sentiment_metrics['total_articles']}
Average Positive Prob   : {sentiment_metrics['avg_positive']*100:.2f} %
Average Neutral Prob    : {sentiment_metrics['avg_neutral']*100:.2f} %
Average Negative Prob   : {sentiment_metrics['avg_negative']*100:.2f} %
Aggregate Sentiment     : {sentiment_metrics['aggregate_score']:+.2f} ({sentiment_metrics['qualitative_label']})

4. NEWS SUMMARY
------------------------------------------------------
{news_summary}

5. OVERALL MARKET CONTEXT & CONFLIC ANALYSIS
------------------------------------------------------
Overall Context Badge   : {market_context['context_badge']}
Market Context Score    : {market_context['market_context_score']} / 100
Status Category         : {market_context['status_category']}
Explanation             : {market_context['explanation']}

======================================================
DISCLAIMER: Educational analysis only. Historical strategy 
performance and news sentiment do not guarantee future market 
returns. This application does NOT provide financial advice.
======================================================
"""
            st.download_button(
                "📄 Download NLP & Market Analysis Report TXT",
                full_nlp_report,
                file_name=f"{ticker}_nlp_market_analysis_report.txt",
                mime="text/plain",
                use_container_width=True
            )

# ======================================================
# ABOUT TAB
# ======================================================

with about_tab:

    st.subheader("NLP-Enhanced Stock Strategy Analysis using Financial News Sentiment and Historical Backtesting")

    st.markdown("""
### 📌 Project Objective

Traditional stock backtesting systems analyze historical price data to identify strategies that performed well in the past, 
but they do not consider the latest financial news and textual information affecting the current market.

This project extends an existing stock-market strategy backtester by collecting current financial news, 
preprocessing the text using NLP techniques, performing financial sentiment analysis (FinBERT) and news summarization (BART/T5), 
and comparing these results with current technical indicators and historical strategy performance to provide an explainable view of the current market context.

> **Important:** The system does not provide financial advice or guarantee future returns. It presents historical evidence and current market/news information so that the user can make their own decision.

---

### 🏗 Architecture & Core Flow

1. **Historical Backtesting:** 
   - Historical OHLC Stock Data from Yahoo Finance
   - 20-Day & 50-Day Moving Average Crossover Strategy
   - Calculates Strategy Return, Buy & Hold Return, Outperformance, Portfolio Curve.

2. **Current Technical Analysis:**
   - Current Stock Price, MA20, MA50
   - Evaluates current trend (MA20 > MA50 → Positive, MA20 < MA50 → Negative).

3. **Current News NLP Pipeline:**
   - Financial news collected via REST API (Finnhub / NewsAPI / GNews with Demo Mode fallback)
   - Classical Preprocessing: Text Cleaning → Tokenization → Stopword Removal → Lemmatization (NLTK)
   - FinBERT Sentiment Classification (ProsusAI/finbert) → Positive / Neutral / Negative Probabilities
   - Aggregate Sentiment Score = Positive % - Negative %
   - BART/T5 Neural Summarization → Executive Financial News Summary.

4. **Integrated Market Context & Signal Conflict Detection:**
   - Combines Historical Performance (40%), Current Technical Condition (30%), and News Sentiment (30%)
   - Computes **Market Context Score (0 to 100)**
   - Detects Signal Agreement / Conflict (🟢 Positive Context, 🟡 Mixed Context, 🔴 Negative Context)

---

### 🔬 Model Performance & Transparency Note

Pretrained **FinBERT** (`ProsusAI/finbert`) inference and **BART** summarization are used for natural language understanding. 
Model performance metrics (Accuracy/Precision/Recall/F1) are not independently evaluated on custom labeled datasets in this release.

---

### 🛠 Technologies Used

Python · Streamlit · Pandas · NumPy · Matplotlib · PyTorch · Hugging Face Transformers (FinBERT, BART/T5) · NLTK · yfinance · Requests
""")