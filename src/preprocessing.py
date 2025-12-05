from pathlib import Path
from typing import Tuple, List, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from .db_utils import load_table

# === Definitive feature sets ===

# CLASSIFICATION: predict orbital_band
NUM_FEATURES_CLASSIF: List[str] = [
    "inclination",
    "eccentricity",
    "launch_year_estimate",
    "days_in_orbit_estimate",
]

CAT_FEATURES_CLASSIF: List[str] = [
    "object_type",
    "satellite_constellation",
    "congestion_risk",
    "orbit_lifetime_category",
    "country",
]

CLASS_TARGET: str = "orbital_band"


# REGRESSION: predict period_minutes
NUM_FEATURES_REG: List[str] = [
    "inclination",
    "eccentricity",
    "launch_year_estimate",
    "days_in_orbit_estimate",
]

CAT_FEATURES_REG: List[str] = [
    "object_type",
    "satellite_constellation",
    "congestion_risk",
    "orbit_lifetime_category",
    "country",
    "orbital_band", 
]

REG_TARGET: str = "period_minutes"


def load_clean_data(db_path: Union[str, Path], table_name: str = "satellites_clean") -> pd.DataFrame:
    """
    Load clean table from dataset
    """
    return load_table(db_path, table_name)


def get_classification_feature_columns() -> Tuple[List[str], List[str]]:
    """
    Gives features (num_features, cat_features) for classification.
    """
    return NUM_FEATURES_CLASSIF, CAT_FEATURES_CLASSIF


def get_regression_feature_columns() -> Tuple[List[str], List[str]]:
    """
    Gives (num_features, cat_features) for regression.
    """
    return NUM_FEATURES_REG, CAT_FEATURES_REG


def prepare_classification_data(df: pd.DataFrame):
    """
    Select X and target y for classification and removes rows with NaN in the relevant columns
    """
    num_cols, cat_cols = get_classification_feature_columns()
    cols = num_cols + cat_cols + [CLASS_TARGET]

    # filter interesting columns
    df_sel = df[cols].copy()

    # Remove NaN rows from feature or target
    df_sel = df_sel.dropna(subset=cols)

    X = df_sel[num_cols + cat_cols]
    y = df_sel[CLASS_TARGET]

    return X, y


def prepare_regression_data(df: pd.DataFrame):
    """
     Select X and target y for regression and removes rows with NaN in the relevant columns
    """
    num_cols, cat_cols = get_regression_feature_columns()
    cols = num_cols + cat_cols + [REG_TARGET]

    df_sel = df[cols].copy()
    df_sel = df_sel.dropna(subset=cols)

    X = df_sel[num_cols + cat_cols]
    y = df_sel[REG_TARGET]

    return X, y


def train_test_split_classification(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train/test split per la classification (with stratify).
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def train_test_split_regression(
    X,
    y,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Train/test split for regression.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )