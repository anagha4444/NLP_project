import re
import nltk

# Safely ensure NLTK resources are available
_nltk_downloaded = False

def init_nltk():
    global _nltk_downloaded
    if _nltk_downloaded:
        return
    resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet')
    ]
    for res_path, res_name in resources:
        try:
            nltk.data.find(res_path)
        except LookupError:
            try:
                nltk.download(res_name, quiet=True)
            except Exception:
                pass
    _nltk_downloaded = True


def clean_text(text: str) -> str:
    """
    Cleans raw financial text.
    Removes URLs, HTML tags, and excessive punctuation/whitespace.
    """
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    # Normalize extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_text(text: str) -> list[str]:
    """Tokenizes text using NLTK or regex fallback."""
    init_nltk()
    cleaned = clean_text(text)
    try:
        tokens = nltk.word_tokenize(cleaned)
    except Exception:
        # Regex fallback tokenizer if NLTK data issue occurs
        tokens = re.findall(r'\b\w+\b', cleaned)
    return tokens


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Removes standard NLTK English stopwords."""
    init_nltk()
    try:
        stop_words = set(nltk.corpus.stopwords.words('english'))
    except Exception:
        # Minimal built-in stopword fallback
        stop_words = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "being", "in", "on", "at", "to", "for", "with", "by", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once"}
    
    return [t for t in tokens if t.lower() not in stop_words and len(t) > 1]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatizes tokens using WordNetLemmatizer."""
    init_nltk()
    try:
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return [lemmatizer.lemmatize(t.lower()) for t in tokens]
    except Exception:
        return [t.lower() for t in tokens]


def preprocess_article(headline: str, description: str = "") -> dict:
    """
    Builds two representations of the article text:
    
    Representation A (Demonstration Pipeline):
        raw_text -> cleaned_text -> raw_tokens -> filtered_tokens -> lemmatized_tokens -> lemmatized_text
        
    Representation B (FinBERT Input):
        clean natural language readable text preserving full context.
    """
    raw_text = f"{headline}. {description}".strip() if description else headline.strip()
    
    # Representation B: Clean readable natural language for transformer model
    representation_b = clean_text(raw_text)
    
    # Representation A: Classic NLP Demonstration Pipeline
    cleaned_lower = representation_b.lower()
    raw_tokens = tokenize_text(cleaned_lower)
    tokens_no_stopwords = remove_stopwords(raw_tokens)
    lemmatized_tokens = lemmatize_tokens(tokens_no_stopwords)
    lemmatized_text = " ".join(lemmatized_tokens)
    
    return {
        "raw_text": raw_text,
        "cleaned_text": representation_b,
        "raw_tokens": raw_tokens,
        "tokens_no_stopwords": tokens_no_stopwords,
        "lemmatized_tokens": lemmatized_tokens,
        "representation_a": lemmatized_text,
        "representation_b": representation_b
    }
