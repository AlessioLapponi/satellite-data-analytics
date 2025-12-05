"""
Train regression models to predict period_minutes.
Saves trained pipelines into models/sklearn/.
"""

from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.preprocessing import (
    load_clean_data,
    prepare_regression_data,
    get_regression_feature_columns,
    train_test_split_regression
)
from src.features import build_preprocessor


def main():
    ROOT = Path(__file__).resolve().parents[1]
    DB_PATH = ROOT / "data" / "processed" / "satellites.db"
    MODELS_DIR = ROOT / "models" / "sklearn"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_clean_data(DB_PATH, table_name="satellites_clean")
    X, y = prepare_regression_data(df)
    num_cols, cat_cols = get_regression_feature_columns()

    X_train, X_test, y_train, y_test = train_test_split_regression(X, y)

    preprocessor = build_preprocessor(num_cols, cat_cols)

    models = {
        "linreg": LinearRegression(),
        "rf_reg": RandomForestRegressor(n_estimators=200, random_state=42),
        "gb_reg": GradientBoostingRegressor(random_state=42),
    }

    for name, reg in models.items():
        print(f"\n=== Training {name} ===")
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", reg)
        ])

        pipe.fit(X_train, y_train)

        # Evaluate
        y_pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        r2 = r2_score(y_test, y_pred)

        print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

        # Save
        model_path = MODELS_DIR / f"regression_{name}.joblib"
        joblib.dump(pipe, model_path)
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()