import sys
import joblib
import pandas as pd
from .utils import clean_text, extract_numeric_features

def predict_text(model_path="models/spam_model.joblib", text=None):
    if text is None:
        print("❌ Please provide text to classify")
        return

    model = joblib.load(model_path)

    clean = clean_text(text)
    df = pd.DataFrame({"text": [text], "clean_text": [clean]})

    num_features = extract_numeric_features(df, "text")

    X = pd.concat(
        [pd.Series(clean, name="clean_text"), num_features.reset_index(drop=True)],
        axis=1,
    )

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]

    print("🔮 Prediction:", "SPAM" if pred == 1 else "NOT SPAM")
    print("📈 Probabilities:", prob)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        predict_text(text=text)
    else:
        print('Usage: python -m src.predict "your message here"')
