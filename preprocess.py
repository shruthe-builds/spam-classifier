"""
preprocess.py
-------------
Text cleaning / NLP preprocessing utilities used by both training and
prediction so the exact same steps are applied every time.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data (only happens once, then it's cached)
for pkg in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}") if "punkt" in pkg else nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    """
    Lowercases, strips URLs/emails/numbers/punctuation, removes stopwords,
    and stems each word. Returns a single cleaned string ready for
    vectorization (e.g. TF-IDF).
    """
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove numbers
    text = re.sub(r"\d+", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [STEMMER.stem(w) for w in words if w not in STOPWORDS and len(w) > 1]

    return " ".join(words)
