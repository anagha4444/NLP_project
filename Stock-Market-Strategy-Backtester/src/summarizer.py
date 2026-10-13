import logging
import re
import pandas as pd
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

logger = logging.getLogger(__name__)

SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"


@st.cache_resource(show_spinner=False)
def load_summarizer_model():
    """
    Loads pretrained Seq2Seq summarization model and tokenizer directly.
    Cached via Streamlit resource manager.
    Returns (tokenizer, model, is_loaded, error_message).
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)
        model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL)
        return tokenizer, model, True, ""
    except Exception as e:
        err_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Failed to load summarizer model: {err_msg}")
        return None, None, False, err_msg


def clean_article_text(headline: str, description: str = "") -> str:
    """
    Cleans headline and description:
    - Removes placeholder boilerplate: 'Recent market developments reported by <source>.'
    - Drops description if it just duplicates the headline.
    - Normalizes excessive whitespace while preserving original casing.
    """
    h = str(headline).strip() if headline else ""
    d = str(description).strip() if description else ""
    
    # Remove 'Recent market developments reported by ...'
    d = re.sub(r'Recent market developments reported by [^.]*\.?', '', d, flags=re.IGNORECASE).strip()
    
    # Normalize internal whitespace
    d = re.sub(r'\s+', ' ', d).strip()
    h = re.sub(r'\s+', ' ', h).strip()
    
    # If description duplicates the headline, discard it
    if d.strip('.').lower() == h.strip('.').lower():
        d = ""
        
    if h and d:
        return f"{h}. {d}"
    return h or d


def fallback_extractive_summary(articles_df: pd.DataFrame) -> str:
    """
    Improved extractive fallback summary:
    - Picks up to 3 distinct headlines (deduplicated).
    - Preserves original capitalization (no lowercasing).
    - Avoids repetition and eliminates placeholder boilerplate.
    """
    if articles_df is None or articles_df.empty:
        return "No recent news available to generate a summary."
        
    seen = set()
    distinct_headlines = []
    
    for _, row in articles_df.iterrows():
        raw_h = row.get("headline")
        if not raw_h or not isinstance(raw_h, str):
            continue
        h = raw_h.strip()
        clean_key = re.sub(r'\W+', '', h).lower()
        if clean_key and clean_key not in seen:
            seen.add(clean_key)
            distinct_headlines.append(h)
        if len(distinct_headlines) == 3:
            break
            
    if not distinct_headlines:
        return "Recent market news reflects ongoing corporate and sector developments."
        
    if len(distinct_headlines) == 1:
        return f'Market reports highlight: "{distinct_headlines[0]}".'
    elif len(distinct_headlines) == 2:
        return f'Key market developments highlight: "{distinct_headlines[0]}", along with: "{distinct_headlines[1]}".'
    else:
        return (
            f'Key market developments highlight: "{distinct_headlines[0]}", '
            f'followed by: "{distinct_headlines[1]}", and: "{distinct_headlines[2]}".'
        )


def generate_news_summary(articles_df: pd.DataFrame, max_articles: int = 5) -> tuple[str, bool]:
    """
    Generates a concise financial news summary from recent news articles.
    Uses AutoTokenizer and AutoModelForSeq2SeqLM directly with model.generate().
    Returns (summary_text, is_ai_generated).
    """
    if articles_df is None or articles_df.empty:
        return "No articles available for news summarization.", False
        
    tokenizer, model, is_loaded, error_msg = load_summarizer_model()
    
    # Collect clean texts from top distinct articles
    seen = set()
    cleaned_snippets = []
    for idx, row in articles_df.iterrows():
        h = row.get("headline", "")
        d = row.get("description", "")
        cleaned = clean_article_text(h, d)
        norm_key = re.sub(r'\W+', '', h).lower()
        if norm_key and norm_key not in seen:
            seen.add(norm_key)
            if cleaned:
                cleaned_snippets.append(cleaned)
        if len(cleaned_snippets) >= max_articles:
            break
            
    if not cleaned_snippets:
        fallback_text = fallback_extractive_summary(articles_df)
        return f"[Headline Summary (AI model unavailable)]: {fallback_text}", False
        
    full_text = " ".join(cleaned_snippets).strip()
    
    # If text is very short (< 15 words), fallback is more appropriate
    if len(full_text.split()) < 15:
        fallback_text = fallback_extractive_summary(articles_df)
        return f"[Headline Summary (AI model unavailable)]: {fallback_text}", False
        
    if is_loaded and model is not None and tokenizer is not None:
        try:
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                max_length=1024,
                truncation=True
            )
            summary_ids = model.generate(
                inputs["input_ids"],
                num_beams=4,
                min_length=30,
                max_length=110,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            res_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            
            try:
                if hasattr(st, "session_state"):
                    st.session_state["is_ai_summary"] = True
                    st.session_state["summarizer_error"] = ""
            except Exception:
                pass
                
            return f"[AI-Generated Summary]: {res_text}", True
            
        except Exception as e:
            gen_error = f"{type(e).__name__}: {str(e)}"
            logger.warning(f"Summarizer generation failed: {gen_error}. Using fallback.")
            try:
                if hasattr(st, "session_state"):
                    st.session_state["is_ai_summary"] = False
                    st.session_state["summarizer_error"] = gen_error
            except Exception:
                pass
    else:
        try:
            if hasattr(st, "session_state"):
                st.session_state["is_ai_summary"] = False
                st.session_state["summarizer_error"] = error_msg
        except Exception:
            pass
            
    fallback_text = fallback_extractive_summary(articles_df)
    return f"[Headline Summary (AI model unavailable)]: {fallback_text}", False
