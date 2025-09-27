import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Cleans text: lowercase, remove punctuation/numbers, remove stopwords."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)   # ✅ I added \s so spaces remain
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text
