import pandas as pd
import numpy as np


def analyze_current_technical_condition(df: pd.DataFrame) -> dict:
    """
    Evaluates current technical indicators from latest available stock data.
    Uses MA20 vs MA50 moving average relationship.
    """
    if df is None or df.empty or "MA20" not in df.columns or "MA50" not in df.columns:
        return {
            "current_price": 0.0,
            "ma20": 0.0,
            "ma50": 0.0,
            "condition_label": "Unknown",
            "is_positive": False,
            "explanation": "Insufficient price data to evaluate moving averages."
        }
        
    latest_row = df.dropna(subset=["Close", "MA20", "MA50"]).iloc[-1]
    price = float(latest_row["Close"])
    ma20 = float(latest_row["MA20"])
    ma50 = float(latest_row["MA50"])
    
    if ma20 > ma50:
        label = "Positive Trend"
        is_pos = True
        explanation = f"MA20 (₹{ma20:,.2f}) is above MA50 (₹{ma50:,.2f}), signaling a positive technical momentum."
    elif ma20 < ma50:
        label = "Negative Trend"
        is_pos = False
        explanation = f"MA20 (₹{ma20:,.2f}) is below MA50 (₹{ma50:,.2f}), signaling a negative technical trend."
    else:
        label = "Neutral Trend"
        is_pos = False
        explanation = f"MA20 (₹{ma20:,.2f}) equals MA50 (₹{ma50:,.2f}), indicating a neutral trend."
        
    return {
        "current_price": price,
        "ma20": ma20,
        "ma50": ma50,
        "condition_label": label,
        "is_positive": is_pos,
        "explanation": explanation
    }


def compute_market_context_score(
    historical_summary: dict,
    technical_info: dict,
    sentiment_metrics: dict,
    weights: dict = None
) -> dict:
    """
    Computes an integrated Market Context Score (0 to 100) and detects signal agreement/conflict.
    
    Weights (default):
    - Historical Performance: 40%
    - Current Technical Condition: 30%
    - Current News Sentiment: 30%
    """
    if weights is None:
        weights = {"historical": 0.40, "technical": 0.30, "sentiment": 0.30}
        
    # 1. Historical Score (0 to 100)
    strat_ret = historical_summary.get("strategy_return", 0.0)
    outperform = historical_summary.get("outperformance", 0.0)
    # Map return to 0-100 scale (50 is neutral 0% return)
    hist_raw = 50.0 + (strat_ret * 0.8) + (outperform * 0.4)
    hist_score = max(0.0, min(100.0, hist_raw))
    hist_pos = strat_ret > 0 and outperform >= -2.0
    
    # 2. Technical Score (0 to 100)
    tech_pos = technical_info.get("is_positive", False)
    price = technical_info.get("current_price", 0.0)
    ma20 = technical_info.get("ma20", 1.0)
    ma50 = technical_info.get("ma50", 1.0)
    
    if ma50 > 0:
        ma_diff_pct = ((ma20 - ma50) / ma50) * 100.0
        tech_raw = 50.0 + (ma_diff_pct * 5.0)
    else:
        tech_raw = 50.0 if not tech_pos else 75.0
        
    tech_score = max(0.0, min(100.0, tech_raw))
    
    # 3. Sentiment Score (0 to 100)
    agg_sent_score = sentiment_metrics.get("aggregate_score", 0.0)  # Range -1 to +1
    sent_score = max(0.0, min(100.0, 50.0 + (agg_sent_score * 50.0)))
    sent_pos = agg_sent_score >= 0.10
    sent_neg = agg_sent_score <= -0.10
    
    # Combined Market Context Score
    total_score = (
        (hist_score * weights["historical"]) +
        (tech_score * weights["technical"]) +
        (sent_score * weights["sentiment"])
    )
    total_score = round(total_score, 1)
    
    # Signal Agreement / Conflict Analysis
    pos_count = sum([1 for flag in [hist_pos, tech_pos, sent_pos] if flag])
    neg_count = sum([1 for flag in [not hist_pos, not tech_pos, sent_neg] if flag])
    
    if hist_pos and tech_pos and sent_pos:
        context_badge = "🟢 Positive Context"
        status_category = "All Signals Positive"
        explanation = (
            "Historical strategy performance, current moving-average technical conditions, "
            "and recent financial news sentiment are broadly positive. All three information sources are currently aligned."
        )
    elif not hist_pos and not tech_pos and sent_neg:
        context_badge = "🔴 Negative Context"
        status_category = "All Signals Negative"
        explanation = (
            "Historical strategy performance was weak, current technical indicators show negative momentum, "
            "and recent financial news sentiment is predominantly negative. Signals are aligned negatively."
        )
    elif hist_pos and (not tech_pos or sent_neg):
        context_badge = "🟡 Mixed / Uncertain Context"
        status_category = "Historical/Current Divergence"
        explanation = (
            "Historical/current divergence detected: Although the strategy performed positively over the selected "
            "historical period, current technical conditions and/or recent financial news sentiment are weaker."
        )
    elif not hist_pos and (tech_pos or sent_pos):
        context_badge = "🟡 Mixed / Uncertain Context"
        status_category = "Historical Weakness vs Current Recovery"
        explanation = (
            "Divergence detected: Historical strategy performance was weak, but current technical trends "
            "and/or financial news sentiment show signs of positive momentum."
        )
    else:
        context_badge = "🟡 Mixed / Uncertain Context"
        status_category = "Mixed Signals"
        explanation = (
            "Market indicators present mixed signals. Historical performance, technical moving averages, "
            "and news sentiment show conflicting directions."
        )
        
    hist_interp = (
        f"Strategy Return: {strat_ret:+.2f}% | Outperformance vs Buy & Hold: {outperform:+.2f}%. "
        + ("Historical strategy performance was stronger than buy-and-hold during the selected period." if outperform > 0 else "Buy-and-hold outperformed the strategy historically.")
    )
    
    tech_interp = technical_info.get("explanation", "")
    
    pos_pct = sentiment_metrics.get("avg_positive", 0.0) * 100
    neu_pct = sentiment_metrics.get("avg_neutral", 0.0) * 100
    neg_pct = sentiment_metrics.get("avg_negative", 0.0) * 100
    sent_interp = f"News Sentiment: {pos_pct:.1f}% Positive, {neu_pct:.1f}% Neutral, {neg_pct:.1f}% Negative. ({sentiment_metrics.get('qualitative_label', '')})"
    
    comparison_table = [
        {
            "Component": "Historical Performance",
            "Current Result": f"Strategy {strat_ret:+.2f}% (Outperf: {outperform:+.2f}%)",
            "Status": "Positive" if hist_pos else "Weak / Negative",
            "Interpretation": hist_interp
        },
        {
            "Component": "Technical Condition",
            "Current Result": f"Price ₹{price:,.2f} | MA20: ₹{ma20:,.2f} | MA50: ₹{ma50:,.2f}",
            "Status": "Positive" if tech_pos else "Negative",
            "Interpretation": tech_interp
        },
        {
            "Component": "News Sentiment",
            "Current Result": f"{pos_pct:.0f}% Pos / {neu_pct:.0f}% Neu / {neg_pct:.0f}% Neg",
            "Status": sentiment_metrics.get('qualitative_label', 'Neutral'),
            "Interpretation": sent_interp
        }
    ]
    
    return {
        "market_context_score": total_score,
        "context_badge": context_badge,
        "status_category": status_category,
        "explanation": explanation,
        "comparison_table": pd.DataFrame(comparison_table),
        "disclaimer": "Educational analysis only. Historical performance and sentiment do not guarantee future market performance. The system does not provide personalized financial advice."
    }
