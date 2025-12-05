"""
Utilities for loading and running predictions using 
trained sklearn and Keras models.
"""

from pathlib import Path
import joblib
import numpy as np
from tensorflow import keras


ROOT = Path(__file__).resolve().parents[1]


# ===============================
#   SKLEARN MODELS
# ===============================

def load_sklearn_model(name: str):
    """
    Load a sklearn model stored in models/sklearn/.
    Example: load_sklearn_model("classification_rf")
    """
    model_path = ROOT / "models" / "sklearn" / f"{name}.joblib"
    return joblib.load(model_path)


def predict_sklearn(model, X_raw):
    """
    Given a loaded sklearn pipeline, make predictions.
    """
    return model.predict(X_raw)


# ===============================
#   KERAS MODELS
# ===============================

def load_keras_model():
    model_path = ROOT / "models" / "keras" / "keras_classification_model.h5"
    preprocessor_path = ROOT / "models" / "keras" / "preprocessor.joblib"
    class_labels_path = ROOT / "models" / "keras" / "class_labels.npy"

    model = keras.models.load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    class_labels = np.load(class_labels_path)

    return model, preprocessor, class_labels


def predict_keras(model, preprocessor, class_labels, X_raw):
    """
    Preprocess raw input and return label string.
    """
    X_proc = preprocessor.transform(X_raw)
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    X_proc = X_proc.astype("float32")

    proba = model.predict(X_proc)
    idx = proba.argmax(axis=1)
    return class_labels[idx]