import streamlit as st
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import joblib
classifiers = joblib.load('classifiers.pkl')
label_encoders = joblib.load('encoders.pkl')


LEVELS = [
    ('Product',     'Level 1'),
    ('Sub-product', 'Level 2'),
    ('Issue',       'Level 3'),
    ('Sub-issue',   'Level 4'),
]

def preprocess_text(text, remove_stopwords=True):
    """
    Full NLP preprocessing pipeline:
    lowercase → remove special chars → tokenize → stopword removal → lemmatization
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove special characters, keep only alphanumeric + spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    
    # 3. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 4. Tokenize
    tokens = word_tokenize(text)
    
    # 5. Stopword removal
    if remove_stopwords:
        tokens = [t for t in tokens if t not in ALL_STOPWORDS and len(t) > 2]
    
    # 6. Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return ' '.join(tokens)


# ── 1. NLTK English stopwords ─────────────────────────────────────────────
base_stopwords = set(stopwords.words('english'))

# ── 2. CFPB dataset specific masking tokens ───────────────────────────────
masking_tokens = {
    'xx', 'xxx', 'xxxx', 'xxxxx', 'xxxxxx', 'xxxxxxx',
    'xxxxxxxx', 'xxxxxxxxx', 'xxxxxxxxxx', 'xxxxxxxxxxx', 'xxxxxxxxxxxx' 
}

# ── 3. Financial domain stopwords ─────────────────────────────────────────
financial_stopwords = {
    # Generic complaint words (no discriminative power)
    'company', 'account', 'would', 'also', 'said', 'told', 'call',
    'called', 'get', 'got', 'one', 'us', 'made', 'make', 'time',
    'since', 'even', 'still', 'back', 'know', 'many', 'could',
    'contact', 'contacted', 'send', 'sent', 'received', 'receive',
    # Regulatory boilerplate
    'cfpb', 'consumer', 'complaint', 'financial', 'bureau',
    'protection', 'federal', 'act', 'law',
    # Date/number noise
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    'month', 'year', 'day', 'date',
    # Generic financial terms too common to distinguish
    'bank', 'money', 'payment', 'paid', 'pay', 'amount',
    'letter', 'information', 'number', 'please',
}

# ── 4. Combine all stopwords ───────────────────────────────────────────────
ALL_STOPWORDS = base_stopwords | masking_tokens | financial_stopwords

print(f"Base NLTK stopwords   : {len(base_stopwords):4d}")
print(f"Masking tokens        : {len(masking_tokens):4d}")
print(f"Financial stopwords   : {len(financial_stopwords):4d}")
print(f"Total unique stopwords: {len(ALL_STOPWORDS):4d}")

# ── 5. Text preprocessing function ────────────────────────────────────────
lemmatizer = WordNetLemmatizer()


# ── 1. Company response to consumer Score mapping ───────────────────────
COMPANY_RESPONSE_SCORE = {
    'Untimely response'                 : 6,    # Most serious: Violation of legal duty to respond
    'In progress'                       : 3,    # Still unresolved → indicates ongoing issue
    'Closed with monetary relief'       : 2,    # Monetary compensation → Recognition of actual damages
    'Closed with non-monetary relief'   : 1,    # Non-monetary relief → Partial acknowledgment
    'Closed without relief'             : 1,    # No relief provided → Consumer dissatisfaction possible
    'Closed with relief'                : 1,    # Relief provided after closure (minority)
    'Closed with explanation'           : 0,    # Closure with explanation → Normal handling
    'Closed'                            : 0,    # Simple closure
}

# ── 2. Company public response point mapping ─────────────────────────────────
def score_public_response(val):
    """ Scoring how disadvantageous the company's public stance is to consumers """
    v = str(val).lower()
    if 'disputes the facts' in v:
        return 3   # Denying the fact itself → Most disadvantageous to the consumer
    if "can't verify or dispute" in v:
        return 2   # Unable to verify → Difficult to resolve
    if 'misunderstanding' in v:
        return 1   # Claiming it is a misunderstanding → Implying consumer fault
    if 'isolated error' in v:
        return 1   # Acknowledging a one-time error → Minor
    if 'third party' in v or 'improvement' in v:
        return 1   # Avoiding responsibility or acknowledging room for improvement
    # 'acted appropriately', 'chooses not to provide', NaN → Normal/Neutral
    return 0

# ── 3. Integrated urgency score function ──────────────────────────────────────────────
def compute_urgency_score(row):
    """
    Rule-based urgency score using only 5 response result columns.
    Utilizes only structured metadata without relying on text (narrative).

    Parameters
    ----------
    row : dict or pd.Series
        Must include the following keys:
        - 'Timely response?'
        - 'Consumer disputed?'
        - 'Company response to consumer'
        - 'Company public response'
        - 'Consumer consent provided?'

    Returns
    -------
    int : urgency score (0 ~ 20)
    """
    score = 0

    # ── (A) Timely response? ───────────────────────────────────────────────
    # Mandatory response within 15 days under CFPB regulations. 'No' = Urgent signal amounting to a legal violation.
    timely = str(row.get('Timely response?', 'Yes')).strip()
    if timely == 'No':
        score += 6

    # ── (B) Consumer disputed? ────────────────────────────────────────────
    # Consumer disputed the company's resolution → Need for reprocessing/escalation
    disputed = str(row.get('Consumer disputed?', 'No')).strip()
    if disputed == 'Yes':
        score += 4

    # ── (C) Company response to consumer ──────────────────────────────────
    company_resp = str(row.get('Company response to consumer', '')).strip()
    score += COMPANY_RESPONSE_SCORE.get(company_resp, 0)

    # ── (D) Company public response ───────────────────────────────────────
    public_resp = row.get('Company public response', '')
    score += score_public_response(public_resp)

    # ── (E) Consumer consent provided? ────────────────────────────────────
    # Failure to provide consent → Unclear resolution as the company cannot publicly refute the narrative
    consent = str(row.get('Consumer consent provided?', '')).strip()
    if consent in ('Consent not provided', 'Other'):
        score += 1

    return score

# ── 4. urgency score → 4-level label conversion ────────────────────────────────────────────
def score_to_urgency_level(score):
    """Map numeric score to 4-level urgency label."""
    if score >= 10:
        return 4  # CRITICAL
    elif score >= 6:
        return 3  # HIGH
    elif score >= 2:
        return 2  # MEDIUM
    else:
        return 1  # LOW

URGENCY_LABELS = {
    4: '🔴 CRITICAL',
    3: '🟠 HIGH',
    2: '🟡 MEDIUM',
    1: '🟢 LOW',
}
#-----------------------------------

# Web app UI using Streamlit
st.set_page_config(page_title="NLP Complaint Classifier", layout="wide")

st.title("Customer Complaint Intelligence System")
st.markdown("Analyzes consumer complaint text to predict \"classification level\" and \"urgency\" in real time.")

# Left side bar : model information
with st.sidebar:
    st.header("Model Status")
    for col, level in LEVELS:
        if classifiers.get(col):
            st.success(f"{level} {col}")
        else:
            st.error(f"{level} Not Trained")

# Main screen: input area
st.subheader("Enter Complaint Narrative")
user_input = st.text_area("", 
                         placeholder="e.g., My credit card was charged an annual fee unexpectedly...",
                         height=150)

if st.button("Run Analysis"):
    if user_input.strip() == "":
        st.warning("Please enter the complaint narrative.")
    else:
        # 1. preprocessing
        processed = preprocess_text(user_input)
        
        # column layout for results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.info("Classification Results")
            results_data = []
            
            for col_name, level_name in LEVELS:
                clf = classifiers.get(col_name)
                le = label_encoders.get(col_name)
                
                if clf and le:
                    prob = clf.predict_proba([processed])
                    pred_idx = prob.argmax()
                    confidence = prob.max()
                    label = le.inverse_transform([pred_idx])[0]
                    
                    results_data.append({
                        "Level": level_name,
                        "Category": label,
                        "Confidence": f"{confidence*100:.1f}%"
                    })
            
            # table display
            st.table(pd.DataFrame(results_data))

        with col2:
            st.info("Urgency Analysis")
            
            # urgency score computation using only structured metadata (simulate with dummy data for now)
            pred_product = results_data[0]['Category'] if results_data else "Unknown"
            temp_row = pd.Series({'processed_text': processed, 'Product': pred_product})
            
            u_score = compute_urgency_score(temp_row)
            u_level = score_to_urgency_level(u_score)
            
            # color mapping for urgency levels
            color_map = {"LOW": "green", "MEDIUM": "orange", "HIGH": "red", "CRITICAL": "darkred"}
            display_color = color_map.get(u_level, "blue")
            
            # metric and label display
            st.metric(label="Urgency Score", value=f"{u_score:.2f}")
            st.markdown(f"### Priority: <span style='color:{display_color}'>{u_level}</span>", unsafe_allow_html=True)
            
            # guidance based on urgency level
            st.write("**Recommended Action:**")
            if u_level in ["HIGH", "CRITICAL"]:
                st.warning("I need immediate assignment of a representative and a response within 24 hours.")
            else:
                st.success("Standard response procedure applies. Can be processed within 3 business days.")

        st.divider()
        st.caption(f"Processed Text: {processed}")