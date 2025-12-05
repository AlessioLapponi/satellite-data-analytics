from pathlib import Path
import numpy as np
import random
import joblib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .preprocessing import (
    load_clean_data,
    prepare_classification_data,
    get_classification_feature_columns,
)
from .features import build_preprocessor


def main():
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)

    ROOT = Path(__file__).resolve().parents[1]
    db_path = ROOT / "data" / "processed" / "satellites.db"
    models_dir = ROOT / "models" / "keras"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean_data(db_path, table_name="satellites_clean")
    X, y = prepare_classification_data(df)
    num_cols, cat_cols = get_classification_feature_columns()

    X_train, X_val, y_train_labels, y_val_labels = train_test_split(
        X, y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    class_labels, y_train_int = np.unique(y_train_labels, return_inverse=True)
    label_to_index = {label: idx for idx, label in enumerate(class_labels)}
    y_val_int = np.array([label_to_index[label] for label in y_val_labels])

    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)

    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_val_proc = X_val_proc.toarray()

    X_train_proc = X_train_proc.astype("float32")
    X_val_proc = X_val_proc.astype("float32")

    input_dim = X_train_proc.shape[1]
    n_classes = len(class_labels)

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        X_train_proc,
        y_train_int,
        validation_data=(X_val_proc, y_val_int),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    # Valutazione veloce
    y_val_pred = model.predict(X_val_proc).argmax(axis=1)
    y_val_pred_labels = class_labels[y_val_pred]
    print(classification_report(y_val_labels, y_val_pred_labels))

    # Salvataggi
    model.save(models_dir / "keras_classification_model.h5")
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    np.save(models_dir / "class_labels.npy", class_labels)

    print("Keras model and artifacts saved to:", models_dir)


if __name__ == "__main__":
    main()