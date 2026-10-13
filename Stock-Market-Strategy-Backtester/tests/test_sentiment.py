import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from sentiment_analyzer import (
    fallback_financial_sentiment,
    analyze_news_sentiment,
    analyze_article_sentiment
)


def test_fallback_no_keywords_sums_to_one():
    """Fallback with no financial keywords must sum to exactly 1.0."""
    text = "The quick brown fox jumps over the lazy dog."
    res = fallback_financial_sentiment(text)
    total = res["positive"] + res["neutral"] + res["negative"]
    assert abs(total - 1.0) < 1e-4, f"Probabilities must sum to 1.0, got {total}"
    assert res["label"] == "neutral"


def test_fallback_positive_keywords_sums_to_one():
    """Fallback with positive financial keywords must sum to exactly 1.0."""
    text = "Company reports strong profit and quarterly revenue growth with dividend boost."
    res = fallback_financial_sentiment(text)
    total = res["positive"] + res["neutral"] + res["negative"]
    assert abs(total - 1.0) < 1e-4, f"Probabilities must sum to 1.0, got {total}"
    assert res["label"] == "positive"
    assert res["positive"] > res["negative"]


def test_fallback_negative_keywords_sums_to_one():
    """Fallback with negative financial keywords must sum to exactly 1.0."""
    text = "Firm reports heavy loss, margin decline, and legal lawsuit risk penalty."
    res = fallback_financial_sentiment(text)
    total = res["positive"] + res["neutral"] + res["negative"]
    assert abs(total - 1.0) < 1e-4, f"Probabilities must sum to 1.0, got {total}"
    assert res["label"] == "negative"
    assert res["negative"] > res["positive"]


def test_fallback_mixed_keywords_sums_to_one():
    """Fallback with mixed positive and negative keywords must sum to exactly 1.0."""
    text = "Revenue growth surge reported despite debt burden and margin decline concern."
    res = fallback_financial_sentiment(text)
    total = res["positive"] + res["neutral"] + res["negative"]
    assert abs(total - 1.0) < 1e-4, f"Probabilities must sum to 1.0, got {total}"
    assert 0.0 <= res["positive"] <= 1.0
    assert 0.0 <= res["neutral"] <= 1.0
    assert 0.0 <= res["negative"] <= 1.0


def test_last_day_to_buy_shares_not_classified_positive():
    """
    Headline 'last day to buy shares' should NOT be classified positive
    because 'buy' is ambiguous and was removed from the positive keyword lexicon.
    """
    text = "last day to buy shares"
    res = fallback_financial_sentiment(text)
    assert res["label"] != "positive", (
        f"'last day to buy shares' should not be classified positive! Got label={res['label']}, "
        f"probs={res}"
    )


def test_aggregate_score_equals_avg_pos_minus_avg_neg():
    """
    The corpus aggregate sentiment score must equal avg_positive - avg_negative.
    """
    sample_news = [
        {"headline": "TCS reports strong profit growth", "description": "Good earnings performance."},
        {"headline": "Tech sector faces decline and debt", "description": "Risks mounting."},
        {"headline": "General market trading session update", "description": "Normal session."}
    ]
    df = pd.DataFrame(sample_news)
    _, summary = analyze_news_sentiment(df)
    
    expected_score = summary["avg_positive"] - summary["avg_negative"]
    assert abs(summary["aggregate_score"] - expected_score) < 1e-3, (
        f"Aggregate score ({summary['aggregate_score']}) must equal avg_pos - avg_neg ({expected_score})"
    )
    
    total_avg = summary["avg_positive"] + summary["avg_neutral"] + summary["avg_negative"]
    assert abs(total_avg - 1.0) < 1e-3, f"Average probabilities must sum to 1.0, got {total_avg}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
