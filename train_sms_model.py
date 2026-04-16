import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline


# Load and prepare data
def load_data(path):
    df = pd.read_csv(path, encoding="latin-1")

    # Keep only needed columns
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']

    # Convert labels to numeric
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})

    return df


#  Build ML pipeline
def build_pipeline():
    tfidf = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.95,
        min_df=2
    )

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear"   # stable + supports n_jobs
    )

    pipeline = Pipeline([
        ("tfidf", tfidf),
        ("clf", clf)
    ])

    return pipeline


#  Train, evaluate, save
def main():
    # Load dataset
    df = load_data("spam.csv")

    # Features and labels
    X = df["text"]
    y = df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Build model
    model = build_pipeline()

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # Save model
    os.makedirs("model_artifacts", exist_ok=True)
    model_path = os.path.join("model_artifacts", "sms_spam_model.joblib")
    joblib.dump(model, model_path)

    print(f"\n Model saved to: {model_path}")


#  Run script
if __name__ == "__main__":
    main()