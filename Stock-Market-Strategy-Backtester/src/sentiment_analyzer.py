import logging
import re
import pandas as pd
import numpy as np
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

logger = logging.getLogger(__name__)

MODEL_NAME = "ProsusAI/finbert"


@st.cache_resource(show_spinner=False)
def load_finbert_pipeline():
    """
    Loads pretrained FinBERT model and tokenizer.
    Cached via Streamlit resource manager.
    Returns (pipeline, is_loaded, error_message).
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        nlp_pipe = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            top_k=None
        )
        return nlp_pipe, True, ""
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Failed to load FinBERT pipeline: {err_msg}")
        return None, False, err_msg


def fallback_financial_sentiment(text: str) -> dict:
    """
    Fallback financial sentiment classifier using a financial domain lexicon.
    Used if FinBERT model fails to load or offline.
    Probabilities are guaranteed to sum to exactly 1.0.
    """
    text_lower = text.lower() if isinstance(text, str) else ""
    
    # Financial domain words (ambiguous words removed: 'record', 'high', 'buy', 'sell', 'cut')
    pos_words = {
        "growth", "profit", "revenue", "surge", "gain", "upgrade",
        "outperform", "bullish", "expand", "dividend", "strong",
        "beat", "positive", "success", "contract", "boost", "buoyant"
    }
    neg_words = {
        "loss", "decline", "fall", "drop", "downgrade", "bearish",
        "pressure", "concern", "risk", "debt", "uncertainty",
        "plunge", "slash", "negative", "investigation", "penalty",
        "lawsuit", "warn"
    }
    
    # Whole-word regex matching
    pos_count = sum(1 for w in pos_words if re.search(r'\b' + re.escape(w) + r'\b', text_lower))
    neg_count = sum(1 for w in neg_words if re.search(r'\b' + re.escape(w) + r'\b', text_lower))
    
    total = pos_count + neg_count
    if total == 0:
        pos_prob, neu_prob, neg_prob = 0.20, 0.65, 0.15
    else:
        raw_pos = pos_count / total
        raw_neg = neg_count / total
        # Use 0.65 as multiplier so pos + neg never exceeds 0.95
        pos_prob = 0.15 + raw_pos * 0.65
        neg_prob = 0.15 + raw_neg * 0.65
        neu_prob = max(0.0, 1.0 - pos_prob - neg_prob)
        
        # Normalization step to ensure exact sum of 1.0
        sum_p = pos_prob + neu_prob + neg_prob
        if sum_p > 0:
            pos_prob = pos_prob / sum_p
            neu_prob = neu_prob / sum_p
            neg_prob = neg_prob / sum_p

    pos_prob = round(pos_prob, 4)
    neg_prob = round(neg_prob, 4)
    neu_prob = round(1.0 - pos_prob - neg_prob, 4)
    
    if pos_prob > neu_prob and pos_prob > neg_prob:
        max_label = "positive"
    elif neg_prob > neu_prob and neg_prob > pos_prob:
        max_label = "negative"
    else:
        max_label = "neutral"
    
    return {
        "positive": pos_prob,
        "neutral": neu_prob,
        "negative": neg_prob,
        "label": max_label
    }


def analyze_article_sentiment(text: str, nlp_pipe=None) -> dict:
    """
    Analyzes sentiment for a single article text.
    Returns dictionary with positive, neutral, negative probabilities, final label,
    and execution metadata.
    """
    if nlp_pipe is not None:
        try:
            # Use truncation=True and max_length=512 for transformer input
            res = nlp_pipe(text, truncation=True, max_length=512)
            if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
                scores = res[0]
            elif isinstance(res, list):
                scores = res
            else:
                scores = [res]
            
            prob_dict = {item['label'].lower(): float(item['score']) for item in scores}
            
            pos = prob_dict.get("positive", 0.0)
            neu = prob_dict.get("neutral", 0.0)
            neg = prob_dict.get("negative", 0.0)
            
            # Normalize to guarantee exact 1.0 sum
            total_prob = pos + neu + neg
            if total_prob > 0:
                pos = round(pos / total_prob, 4)
                neg = round(neg / total_prob, 4)
                neu = round(1.0 - pos - neg, 4)
            
            best_label = max(prob_dict, key=prob_dict.get)
            return {
                "positive": pos,
                "neutral": neu,
                "negative": neg,
                "label": best_label,
                "used_finbert": True,
                "error": ""
            }
        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}"
            logger.warning(f"FinBERT inference failed: {err_msg}. Using fallback.")
            fallback_res = fallback_financial_sentiment(text)
            fallback_res["used_finbert"] = False
            fallback_res["error"] = err_msg
            return fallback_res

    fallback_res = fallback_financial_sentiment(text)
    fallback_res["used_finbert"] = False
    fallback_res["error"] = "FinBERT pipeline not loaded"
    return fallback_res


def analyze_news_sentiment(articles_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Computes sentiment probabilities and predictions for all articles in DataFrame.
    Returns updated DataFrame and summary statistics dictionary.
    """
    df = articles_df.copy()
    pipe_res = load_finbert_pipeline()
    if len(pipe_res) == 3:
        nlp_pipe, is_finbert_loaded, load_error = pipe_res
    else:
        nlp_pipe, is_finbert_loaded = pipe_res
        load_error = ""
    
    pos_probs = []
    neu_probs = []
    neg_probs = []
    labels = []
    finbert_success_count = 0
    inference_error = ""
    
    for idx, row in df.iterrows():
        headline = str(row.get("headline") or "").strip()
        description = str(row.get("description") or "").strip()
        # Clean boilerplate filler and drop duplicate descriptions
        clean_d = re.sub(r'Recent market developments reported by [^.]*\.?', '', description, flags=re.IGNORECASE).strip()
        if clean_d.strip('.').lower() == headline.strip('.').lower():
            clean_d = ""
        input_text = f"{headline}. {clean_d}".strip(". ") if clean_d else headline
        res = analyze_article_sentiment(input_text, nlp_pipe)
        
        pos_probs.append(res["positive"])
        neu_probs.append(res["neutral"])
        neg_probs.append(res["negative"])
        labels.append(res["label"].capitalize())
        
        if res.get("used_finbert"):
            finbert_success_count += 1
        elif not inference_error and res.get("error"):
            inference_error = res["error"]
    
    df["positive_probability"] = pos_probs
    df["neutral_probability"] = neu_probs
    df["negative_probability"] = neg_probs
    df["sentiment"] = labels
    df["sentiment_score"] = df["positive_probability"] - df["negative_probability"]
    
    # Determine whether FinBERT was genuinely used
    actual_finbert_used = is_finbert_loaded and (finbert_success_count == len(df)) and (len(df) > 0)
    final_error = load_error or inference_error
    
    # Record in st.session_state if available
    try:
        if hasattr(st, "session_state"):
            st.session_state["is_finbert_used"] = actual_finbert_used
            st.session_state["finbert_error"] = final_error
    except Exception:
        pass
    
    # Summary metrics
    avg_pos = float(np.mean(pos_probs)) if len(pos_probs) > 0 else 0.0
    avg_neu = float(np.mean(neu_probs)) if len(neu_probs) > 0 else 0.0
    avg_neg = float(np.mean(neg_probs)) if len(neg_probs) > 0 else 0.0
    
    # Ensure average probabilities sum to 1.0
    total_avg = avg_pos + avg_neu + avg_neg
    if total_avg > 0:
        avg_pos = round(avg_pos / total_avg, 4)
        avg_neg = round(avg_neg / total_avg, 4)
        avg_neu = round(1.0 - avg_pos - avg_neg, 4)
        
    aggregate_score = round(avg_pos - avg_neg, 4)  # Range -1 to +1
    
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
        "is_finbert_used": actual_finbert_used,
        "finbert_error": final_error
    }
    
    return df, summary_metrics
