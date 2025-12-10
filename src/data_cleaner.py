import pandas as pd
from .utils import clean_text

def load_and_clean(path):
    df = pd.read_csv(path, encoding="latin-1")

    # Remove extra columns
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Rename to label/text
    cols = df.columns.tolist()
    df = df.rename(columns={cols[0]: "label", cols[1]: "text"})

    # Remove nulls
    df = df.dropna(subset=["label", "text"])

    # Encode labels
    df["target"] = df["label"].astype(str).str.lower().map(
        lambda x: 1 if x == "spam" else 0
    )

    df["clean_text"] = df["text"].apply(clean_text)

    return df
