import os
import sys
import pytest
import pandas as pd

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from summarizer import (
    clean_article_text,
    fallback_extractive_summary,
    generate_news_summary
)


def test_cleaning_removes_reported_by_placeholder():
    """Verify that 'Recent market developments reported by ...' is completely stripped."""
    headline = "HCLTech jumps over 7%, TCS up 5%: Why are IT stocks rallying"
    raw_desc = f"{headline}. Recent market developments reported by The Times of India."
    
    cleaned = clean_article_text(headline, raw_desc)
    assert "reported by" not in cleaned.lower()
    assert "Recent market developments" not in cleaned
    assert cleaned == headline


def test_cleaning_drops_duplicate_description():
    """Verify that description is dropped if it just duplicates the headline."""
    headline = "TCS Announces Dividend of Rs 12 per Share"
    duplicate_desc = "TCS Announces Dividend of Rs 12 per Share."
    
    cleaned = clean_article_text(headline, duplicate_desc)
    assert cleaned == headline


def test_cleaning_keeps_valid_non_duplicate_description():
    """Verify that genuine descriptions are retained and appended."""
    headline = "TCS Reports Strong Q4 Results"
    real_desc = "Net profit rose 9 percent driven by robust enterprise cloud transformation demand."
    
    cleaned = clean_article_text(headline, real_desc)
    assert headline in cleaned
    assert real_desc in cleaned
    assert cleaned == f"{headline}. {real_desc}"


def test_fallback_summary_no_duplicates_and_preserves_case():
    """
    Verify fallback extractive summary:
    1. Deduplicates repeating headlines.
    2. Preserves original title capitalization (does NOT force text to lowercase).
    3. Never includes boilerplate 'reported by' sentences.
    """
    news_items = [
        {"headline": "HCLTech Jumps Over 7%, TCS Up 5%", "description": "Duplicate story 1."},
        {"headline": "HCLTech Jumps Over 7%, TCS Up 5%", "description": "Duplicate story 2."},
        {"headline": "TCS, Vedanta Dividends: Last Day To Buy Shares", "description": "Dividend news."},
        {"headline": "India's TCS Rises As Margins Steady", "description": "Margin news."}
    ]
    df = pd.DataFrame(news_items)
    summary_text = fallback_extractive_summary(df)
    
    # Check no duplicate headline inclusion
    count_first_headline = summary_text.count("HCLTech Jumps Over 7%, TCS Up 5%")
    assert count_first_headline == 1, f"Headline should appear only once, got {count_first_headline}"
    
    # Check original capitalization preserved
    assert "HCLTech Jumps" in summary_text
    assert "hcltech jumps" not in summary_text
    
    # Check no template boilerplate
    assert "reported by" not in summary_text.lower()
    assert "recent operational developments report that" not in summary_text.lower()


def test_generate_news_summary_returns_tuple_and_never_raises():
    """
    Verify generate_news_summary always returns a (str, bool) tuple
    without raising unhandled exceptions across various inputs.
    """
    # Case 1: Empty DataFrame
    res_text, is_ai = generate_news_summary(pd.DataFrame())
    assert isinstance(res_text, str)
    assert isinstance(is_ai, bool)
    assert is_ai is False

    # Case 2: None input
    res_text, is_ai = generate_news_summary(None)
    assert isinstance(res_text, str)
    assert isinstance(is_ai, bool)
    assert is_ai is False

    # Case 3: Valid news items
    news_items = [
        {"headline": "TCS Expands Cloud Deal in Europe", "description": "Multi-million dollar deal."},
        {"headline": "IT Sector Hiring Surges in Q2", "description": "Demand for tech workers grows."}
    ]
    df = pd.DataFrame(news_items)
    res_text, is_ai = generate_news_summary(df)
    assert isinstance(res_text, str)
    assert isinstance(is_ai, bool)
    assert len(res_text) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
