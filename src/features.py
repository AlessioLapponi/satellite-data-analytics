from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def build_preprocessor(num_features: List[str], cat_features: List[str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer with:
      - StandardScaler for numerical features
      - OneHotEncoder for categorical features
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    return preprocessor