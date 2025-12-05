"""
Train classification models to predict orbital_band.
Saves trained pipelines into models/sklearn/.
"""

from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.preprocessing import (
    load_clean_data,
    prepare_classification_data,
    get_classification_feature_columns,
    train_test_split_classification
)
from src.features import build_preprocessor


def main():
    ROOT = Path(__file__).resolve().parents[1]
    DB_PATH = ROOT / "data" / "processed" / "satellites.db"
    MODELS_DIR = ROOT / "models" / "sklearn"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_clean_data(DB_PATH, table_name="satellites_clean")
    X, y = prepare_classification_data(df)
    num_cols, cat_cols = get_classification_feature_columns()

    X_train, X_test, y_train, y_test = train_test_split_classification(X, y)

    preprocessor = build_preprocessor(num_cols, cat_cols)

    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
    }

    for name, clf in models.items():
        print(f"\n=== Training {name} ===")
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", clf)
        ])

        pipe.fit(X_train, y_train)

        # Evaluate
        y_pred = pipe.predict(X_test)
        print(classification_report(y_test, y_pred))

        # Save
        model_path = MODELS_DIR / f"classification_{name}.joblib"
        joblib.dump(pipe, model_path)
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()