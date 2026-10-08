import pandas as pd
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MODEL_NAME = "ProsusAI/finbert"


@st.cache_resource(show_spinner=False)
def load_finbert_pipeline():
    """
    Loads pretrained FinBERT model and tokenizer.
    Cached via Streamlit resource manager.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        nlp_pipe = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True
        )
        return nlp_pipe, True
    except Exception as e:
        return None, False


def fallback_financial_sentiment(text: str) -> dict:
    """
    Fallback financial sentiment classifier using a financial domain lexicon.
    Used if FinBERT model fails to load or offline.
    """
    text_lower = text.lower()
    pos_words = {"growth", "profit", "revenue", "surge", "gain", "upgrade", "outperform", "bullish", "record", "expand", "dividend", "strong", "beat", "positive", "high", "success", "contract", "boost", "buoyant", "buy"}
    neg_words = {"loss", "decline", "fall", "drop", "downgrade", "bearish", "pressure", "concern", "risk", "debt", "uncertainty", "plunge", "slash", "cut", "negative", "investigation", "penalty", "lawsuit", "warn", "sell"}
    
    pos_count = sum(1 for w in pos_words if w in text_lower)
    neg_count = sum(1 for w in neg_words if w in text_lower)
    
    total = pos_count + neg_count
    if total == 0:
        pos_prob, neu_prob, neg_prob = 0.20, 0.65, 0.15
    else:
        raw_pos = pos_count / total
        raw_neg = neg_count / total
        pos_prob = round(0.15 + raw_pos * 0.70, 4)
        neg_prob = round(0.15 + raw_neg * 0.70, 4)
        neu_prob = max(0.05, round(1.0 - pos_prob - neg_prob, 4))
    
    max_label = "positive" if pos_prob > max(neu_prob, neg_prob) else ("negative" if neg_prob > max(pos_prob, neu_prob) else "neutral")
    
    return {
        "positive": pos_prob,
        "neutral": neu_prob,
        "negative": neg_prob,
        "label": max_label
    }


def analyze_article_sentiment(text: str, nlp_pipe=None) -> dict:
    """
    Analyzes sentiment for a single article text.
    Returns dictionary with positive, neutral, negative probabilities and final label.
    """
    if nlp_pipe is not None:
        try:
            # FinBERT expects text truncated to ~512 tokens max
            truncated_text = text[:1000]
            scores = nlp_pipe(truncated_text)[0]
            prob_dict = {item['label'].lower(): float(item['score']) for item in scores}
            
            pos = prob_dict.get("positive", 0.0)
            neu = prob_dict.get("neutral", 0.0)
            neg = prob_dict.get("negative", 0.0)
            
            best_label = max(prob_dict, key=prob_dict.get)
            return {
                "positive": pos,
                "neutral": neu,
                "negative": neg,
                "label": best_label
            }
        except Exception:
            pass
            
    return fallback_financial_sentiment(text)


def analyze_news_sentiment(articles_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Computes sentiment probabilities and predictions for all articles in DataFrame.
    Returns updated DataFrame and summary statistics dictionary.
    """
    df = articles_df.copy()
    nlp_pipe, is_finbert_loaded = load_finbert_pipeline()
    
    pos_probs = []
    neu_probs = []
    neg_probs = []
    labels = []
    
    for idx, row in df.iterrows():
        input_text = row.get("description") or row.get("headline") or ""
        res = analyze_article_sentiment(input_text, nlp_pipe)
        
        pos_probs.append(res["positive"])
        neu_probs.append(res["neutral"])
        neg_probs.append(res["negative"])
        labels.append(res["label"].capitalize())
    
    df["positive_probability"] = pos_probs
    df["neutral_probability"] = neu_probs
    df["negative_probability"] = neg_probs
    df["sentiment"] = labels
    df["sentiment_score"] = df["positive_probability"] - df["negative_probability"]
    
    # Summary metrics
    avg_pos = float(np.mean(pos_probs)) if len(pos_probs) > 0 else 0.0
    avg_neu = float(np.mean(neu_probs)) if len(neu_probs) > 0 else 0.0
    avg_neg = float(np.mean(neg_probs)) if len(neg_probs) > 0 else 0.0
    
    aggregate_score = avg_pos - avg_neg  # Range -1 to +1
    
    pos_count = int(sum(1 for l in labels if l == "Positive"))
    neu_count = int(sum(1 for l in labels if l == "Neutral"))
    neg_count = int(sum(1 for l in labels if l == "Negative"))
    
    if aggregate_score >= 0.35:
        qualitative = "Strongly Positive"
    elif aggregate_score >= 0.10:
        qualitative = "Moderately Positive"
    elif aggregate_score > -0.10:
        qualitative = "Neutral / Balanced"
    elif aggregate_score > -0.35:
        qualitative = "Moderately Negative"
    else:
        qualitative = "Strongly Negative"
        
    summary_metrics = {
        "avg_positive": avg_pos,
        "avg_neutral": avg_neu,
        "avg_negative": avg_neg,
        "aggregate_score": aggregate_score,
        "qualitative_label": qualitative,
        "pos_count": pos_count,
        "neu_count": neu_count,
        "neg_count": neg_count,
        "total_articles": len(df),
        "is_finbert_used": is_finbert_loaded
    }
    
    return df, summary_metrics
