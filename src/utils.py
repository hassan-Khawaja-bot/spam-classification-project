import re
import pandas as pd

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_numeric_features(df, text_col="text"):
    df = df.copy()
    df['msg_len'] = df[text_col].astype(str).apply(len)
    df['num_exclam'] = df[text_col].astype(str).apply(lambda s: s.count("!"))
    df['num_question'] = df[text_col].astype(str).apply(lambda s: s.count("?"))
    df['num_digits'] = df[text_col].astype(str).apply(lambda s: sum(c.isdigit() for c in s))
    df['upper_ratio'] = df[text_col].astype(str).apply(
        lambda s: sum(c.isupper() for c in s) / (len(s) + 1)
    )
    df['num_urls'] = df[text_col].astype(str).apply(lambda s: 1 if re.search(r"http\S+", s) else 0)

    return df[['msg_len', 'num_exclam', 'num_question', 'num_digits', 'upper_ratio', 'num_urls']]
