import pandas as pd
import streamlit as st
from transformers import pipeline

SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"


@st.cache_resource(show_spinner=False)
def load_summarizer_pipeline():
    """
    Loads pretrained summarization model.
    Cached via Streamlit resource manager.
    """
    try:
        summarizer = pipeline(
            "summarization",
            model=SUMMARIZER_MODEL,
            tokenizer=SUMMARIZER_MODEL
        )
        return summarizer, True
    except Exception:
        return None, False


def fallback_extractive_summary(articles_df: pd.DataFrame) -> str:
    """
    Extractive fallback summary when transformer model is offline or uninstalled.
    Synthesizes top news headlines into a clean executive summary.
    """
    if articles_df is None or articles_df.empty:
        return "No recent news available to generate a summary."
        
    headlines = articles_df["headline"].dropna().tolist()[:4]
    descriptions = [d for d in articles_df["description"].dropna().tolist() if len(d) > 20][:3]
    
    parts = []
    if headlines:
        parts.append(f"Key market headlines highlight: '{headlines[0]}'")
        if len(headlines) > 1:
            parts.append(f"along with '{headlines[1]}'.")
    if descriptions:
        parts.append(f"Recent operational developments report that {descriptions[0].lower()}")
        
    summary_text = " ".join(parts)
    if not summary_text:
        summary_text = "Recent market news reflects ongoing corporate developments and macro market observations."
        
    return f"[Key Headline Summary]: {summary_text}"


def generate_news_summary(articles_df: pd.DataFrame, max_articles: int = 5) -> tuple[str, bool]:
    """
    Generates a concise financial news summary from recent news articles.
    Returns (summary_text, is_ai_generated).
    """
    if articles_df is None or articles_df.empty:
        return "No articles available for news summarization.", False
        
    summarizer, is_loaded = load_summarizer_pipeline()
    
    # Combine headlines + descriptions of top articles
    combined_texts = []
    for idx, row in articles_df.head(max_articles).iterrows():
        h = str(row.get("headline", ""))
        d = str(row.get("description", ""))
        combined_texts.append(f"{h}. {d}")
        
    full_text = " ".join(combined_texts).strip()
    
    if len(full_text) < 50:
        return fallback_extractive_summary(articles_df), False
        
    if is_loaded and summarizer is not None:
        try:
            # Truncate input text to fit within model limits (~1024 tokens)
            input_text = full_text[:2000]
            min_len = min(40, max(15, len(input_text.split()) // 3))
            max_len = min(120, max(50, len(input_text.split()) // 2))
            
            summary_res = summarizer(input_text, max_length=max_len, min_length=min_len, do_sample=False)
            res_text = summary_res[0]["summary_text"].strip()
            return f"[AI-Generated Summary]: {res_text}", True
        except Exception:
            pass
            
    return fallback_extractive_summary(articles_df), False
