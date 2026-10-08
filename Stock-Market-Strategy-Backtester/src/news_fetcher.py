import os
import requests
import pandas as pd
import streamlit as st
import xml.etree.ElementTree as ET
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from data_loader import get_company_name

DEMO_NEWS_DATA = {
    "TCS.NS": [
        {
            "headline": "TCS Reports Strong Q4 Earnings with 9% YoY Revenue Growth",
            "source": "Financial Express",
            "publication_date": "2026-10-01",
            "url": "https://example.com/tcs-q4-results",
            "description": "Tata Consultancy Services delivered robust quarterly financial results driven by cloud transformation deals and digital service demand across Europe and North America.",
            "ticker": "TCS.NS"
        },
        {
            "headline": "TCS Expands Strategic AI Partnership with Major Global Enterprise",
            "source": "Economic Times",
            "publication_date": "2026-09-28",
            "url": "https://example.com/tcs-ai-partnership",
            "description": "TCS announced a multi-year expansion of its generative AI suite for enterprise clients, solidifying market leadership in IT services.",
            "ticker": "TCS.NS"
        },
        {
            "headline": "IT Sector Faces Margin Pressure Amid Global Macroeconomic Uncertainty",
            "source": "Business Standard",
            "publication_date": "2026-09-25",
            "url": "https://example.com/it-sector-pressures",
            "description": "Analysts caution that wage hikes and delayed client discretionary spend could temper near-term operating margins across major Indian IT firms.",
            "ticker": "TCS.NS"
        },
        {
            "headline": "TCS Wins $500M Digital Modernization Contract in Europe",
            "source": "LiveMint",
            "publication_date": "2026-09-20",
            "url": "https://example.com/tcs-europe-deal",
            "description": "Tata Consultancy Services secured a landmark multi-year contract to modernize core cloud infrastructure for a European financial institution.",
            "ticker": "TCS.NS"
        },
        {
            "headline": "Brokerages Maintain Buy Rating on TCS Following Steady Order Wins",
            "source": "NDTV Profit",
            "publication_date": "2026-09-15",
            "url": "https://example.com/tcs-brokerage-ratings",
            "description": "Leading brokerages reaffirmed positive long-term outlooks for TCS, citing strong balance sheet resilience and market share gains.",
            "ticker": "TCS.NS"
        }
    ],
    "AAPL": [
        {
            "headline": "Apple Unveils New Product Lineup with Advanced AI Features",
            "source": "Reuters",
            "publication_date": "2026-10-02",
            "url": "https://example.com/apple-new-lineup",
            "description": "Apple announced its latest hardware lineup featuring integrated on-device artificial intelligence processing, driving positive analyst sentiment.",
            "ticker": "AAPL"
        },
        {
            "headline": "Apple Services Revenue Reaches All-Time High in Latest Quarter",
            "source": "Bloomberg",
            "publication_date": "2026-09-29",
            "url": "https://example.com/apple-services-record",
            "description": "App Store, iCloud, and Apple Music growth propelled overall service segment revenues above Wall Street consensus estimates.",
            "ticker": "AAPL"
        },
        {
            "headline": "Supply Chain Adjustments May Briefly Impact Hardware Deliveries",
            "source": "Wall Street Journal",
            "publication_date": "2026-09-24",
            "url": "https://example.com/apple-supply-chain",
            "description": "Minor supply chain adjustments in Asian manufacturing hubs could lead to temporary inventory tightness for flagship devices.",
            "ticker": "AAPL"
        },
        {
            "headline": "Analysts Raise Price Target for Apple Ahead of Holiday Shopping Season",
            "source": "CNBC",
            "publication_date": "2026-09-18",
            "url": "https://example.com/apple-price-target",
            "description": "Strong consumer demand indicators and ecosystem lock-in prompt investment firms to raise price targets for Apple Inc.",
            "ticker": "AAPL"
        },
        {
            "headline": "Regulatory Scrutiny on App Store Practices Continues in Europe",
            "source": "TechCrunch",
            "publication_date": "2026-09-12",
            "url": "https://example.com/apple-eu-regulations",
            "description": "European regulators continue examining digital market compliance, though financial impact remains limited according to analysts.",
            "ticker": "AAPL"
        }
    ]
}


def get_news_api_key():
    """Retrieve API key from Streamlit secrets or environment variables."""
    for key_name in ["FINNHUB_API_KEY", "NEWS_API_KEY", "GNEWS_API_KEY", "NEWS_KEY"]:
        try:
            if key_name in st.secrets:
                return key_name, st.secrets[key_name]
        except Exception:
            pass
        env_val = os.environ.get(key_name)
        if env_val:
            return key_name, env_val
    return None, None


def fetch_gnews_rss(ticker: str, max_articles: int = 15) -> pd.DataFrame | None:
    """
    Fetches live financial news from Google News RSS feed.
    Works for Indian (NSE/BSE), US, and global stock tickers without API key limits.
    """
    try:
        company_name = get_company_name(ticker)
        query = f"{company_name} stock news"
        encoded_query = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        xml_data = urllib.request.urlopen(req, timeout=8).read()
        root = ET.fromstring(xml_data)
        items = root.findall(".//item")
        
        articles = []
        for item in items[:max_articles]:
            raw_title = item.find("title").text if item.find("title") is not None else ""
            pub_date_raw = item.find("pubDate").text if item.find("pubDate") is not None else ""
            link = item.find("link").text if item.find("link") is not None else ""
            
            # Split source from headline if formatted as "Headline - Source"
            source = "Google News"
            headline = raw_title
            if " - " in raw_title:
                parts = raw_title.rsplit(" - ", 1)
                headline = parts[0]
                source = parts[1]
                
            # Parse publication date
            pub_date = datetime.now().strftime("%Y-%m-%d")
            if pub_date_raw:
                try:
                    pub_date = datetime.strptime(pub_date_raw[:16], "%a, %d %b %Y").strftime("%Y-%m-%d")
                except Exception:
                    pass
                    
            articles.append({
                "headline": headline,
                "source": source,
                "publication_date": pub_date,
                "url": link,
                "description": f"{headline}. Recent market developments reported by {source}.",
                "ticker": ticker,
                "is_demo": False
            })
            
        if articles:
            return pd.DataFrame(articles)
    except Exception:
        pass
        
    return None


def get_demo_news(ticker: str) -> pd.DataFrame:
    """Generate realistic demo news dataset for presentation or fallback."""
    ticker_clean = ticker.upper().strip()
    articles = DEMO_NEWS_DATA.get(ticker_clean)
    
    if not articles:
        company = get_company_name(ticker)
        articles = [
            {
                "headline": f"{company} Reports Stable Financial Performance and Growth Outlook",
                "source": "Market Watch",
                "publication_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                "url": f"https://example.com/{ticker}-earnings",
                "description": f"{company} published strong operational metrics with positive revenue trajectory across major business divisions.",
                "ticker": ticker_clean
            },
            {
                "headline": f"Institutional Investors Increase Stake in {company}",
                "source": "Reuters",
                "publication_date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
                "url": f"https://example.com/{ticker}-institutional",
                "description": f"Recent institutional filings show increased portfolio allocation toward {company} following strategic expansion announcements.",
                "ticker": ticker_clean
            },
            {
                "headline": f"Sector Volatility Influences Near-Term Trading Volume for {company}",
                "source": "Bloomberg",
                "publication_date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
                "url": f"https://example.com/{ticker}-volatility",
                "description": f"Broader market macroeconomic uncertainty creates temporary trading fluctuations for {company} stock.",
                "ticker": ticker_clean
            },
            {
                "headline": f"{company} Announces Strategic R&D Expansion and Digital Transformation",
                "source": "Financial Times",
                "publication_date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
                "url": f"https://example.com/{ticker}-rd",
                "description": f"Management outlined major technological investment plans aimed at boosting long-term efficiency and competitiveness.",
                "ticker": ticker_clean
            },
            {
                "headline": f"Analysts Reaffirm Hold/Buy Ratings for {company}",
                "source": "Business Insider",
                "publication_date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                "url": f"https://example.com/{ticker}-ratings",
                "description": f"Equity research reports highlight balanced risk-reward profile for {company} supported by strong fundamentals.",
                "ticker": ticker_clean
            }
        ]
    
    df = pd.DataFrame(articles)
    df["is_demo"] = True
    return df


@st.cache_data(ttl=900, show_spinner=False)
def fetch_financial_news(ticker: str, max_articles: int = 15) -> tuple[pd.DataFrame | None, str]:
    """
    Fetches latest financial news for a ticker.
    Supports Finnhub API for US stocks, Google News RSS for Indian/Global stocks, and Demo Mode as fallback.
    Returns (DataFrame, status_message).
    """
    key_name, api_key = get_news_api_key()
    ticker_symbol = ticker.upper().strip()
    clean_symbol = ticker_symbol.split('.')[0]
    
    # 1. Try Finnhub API if API key is present
    if api_key and key_name == "FINNHUB_API_KEY":
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            url = f"https://finnhub.io/api/v1/company-news?symbol={clean_symbol}&from={from_date}&to={today}&token={api_key}"
            resp = requests.get(url, timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 0:
                    articles = []
                    for item in data[:max_articles]:
                        pub_date = datetime.fromtimestamp(item.get("datetime", 0)).strftime("%Y-%m-%d") if item.get("datetime") else ""
                        articles.append({
                            "headline": item.get("headline", ""),
                            "source": item.get("source", "Finnhub"),
                            "publication_date": pub_date,
                            "url": item.get("url", ""),
                            "description": item.get("summary", ""),
                            "ticker": ticker_symbol,
                            "is_demo": False
                        })
                    return pd.DataFrame(articles), "✅ Successfully fetched live financial news from Finnhub API."
        except Exception:
            pass

    # 2. Try Google News Live RSS Feed (Works for Indian NSE stocks like TCS.NS, RELIANCE.NS & global stocks)
    gnews_df = fetch_gnews_rss(ticker_symbol, max_articles)
    if gnews_df is not None and not gnews_df.empty:
        return gnews_df, f"✅ Successfully fetched live financial news for {ticker_symbol}."

    # 3. Fallback to Demo Mode
    demo_df = get_demo_news(ticker_symbol)
    return demo_df, "⚠️ Showing DEMO MODE financial news fallback."
