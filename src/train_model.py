import os
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from .data_cleaner import load_and_clean
from .utils import extract_numeric_features

def train(
        path="data/spam.csv",
        model_path="models/spam_model.joblib",
        metrics_path="reports/metrics.json",
):

    print("🔄 Loading dataset…")
    df = load_and_clean(path)

    X_text = df["clean_text"]
    X_num = extract_numeric_features(df, "text")

    X = pd.concat(
        [X_text.rename("clean_text"), X_num.reset_index(drop=True)],
        axis=1
    )
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2)), "clean_text"),
            ("num", StandardScaler(), X_num.columns.tolist()),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    param_grid = {
        "clf__C": [0.1, 1.0, 10.0],
    }

    print("🔍 Running GridSearchCV… (3-fold)")
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="f1",
        verbose=1,
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    print("🎯 Best Parameters:", grid.best_params_)

    y_pred = best_model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "best_params": grid.best_params_,
                "classification_report": report,
                "confusion_matrix": cm,
            },
            f,
            indent=2,
        )

    print("✅ Training complete!")
    print("📦 Model saved:", model_path)
    print("📊 Metrics saved:", metrics_path)


if __name__ == "__main__":
    train()
