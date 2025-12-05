# Satellite Orbit Analytics  
**Machine Learning & Deep Learning for Orbital Classification and Period Prediction**

This project applies data science, SQL analytics, machine learning, and deep learning (Keras) to a catalog of Earth-orbiting satellites.

It demonstrates a complete end-to-end workflow:

1. Data ingestion & cleaning (SQL)
2. Exploratory analysis (SQL + pandas + visualizations)
3. Feature engineering
4. ML models (classification + regression)
5. Neural network classifier (Keras)
6. Reusable training scripts & inference utilities


# Project Structure
```
satellite-orbit-analytics/
├─ data/
│  ├─ raw/                      # Original CSV dataset
│  │   └─ satellite_orbital_catalog.csv # to be downloaded separately, if needed
│  └─ processed/
│      └─ satellites.db         # SQLite database with cleaned data
│
├─ sql/
│  ├─ create_tables.sql         # Import raw CSV -> SQLite schema
│  ├─ data_cleaning.sql         # Cleaning, normalization, feature creation
│  └─ exploration_queries.sql   # Data exploration queries (EDA)
│
├─ notebooks/
│  ├─ 01_sql_exploration.ipynb
│  ├─ 02_classification_models.ipynb
│  ├─ 03_regression_models.ipynb
│  └─ 04_keras_neural_network.ipynb
│
├─ src/
│  ├─ db_utils.py
│  ├─ preprocessing.py
│  ├─ features.py
│  ├─ train_classification.py   # Train ML classification
│  ├─ train_regression.py       # Train ML regression
│  └─ train_keras.py            # Train Keras neural network
│  └─ inference.py              # Load/use trained models
│
├─ models/
│  ├─ sklearn/                  # Saved ML models (.joblib)
│  └─ keras/                    # Keras model (.h5) + preprocessor
│
├─ README.md
└─ requirements.txt
```

## Dataset

The project is based on an orbital catalog of Earth satellites (TLE-derived and enriched with additional metadata).

- The **cleaned dataset** is provided directly as:
  - `data/processed/satellites.db`
  - main table: `satellites_clean`

- The **original raw CSV file** (e.g. `satellite_orbital_catalog.csv`) is **not shipped** in this repository for licensing reasons.  
  If you want to fully reproduce the cleaning pipeline from scratch, you can:
  1. Download the original CSV from the data source (e.g. Kaggle / external catalog).
  2. Place it under `data/raw/`.
  3. Use the SQL scripts and run `db_utils.py` to rebuild `satellites.db`.
  
For training and using the models, the included `satellites.db` is sufficient.

The original data source is updated every month, the results shown are based on the observation done December 1st 2025. By redownloading the source data in 2026, and re-create `satellites.db`, similar but different results are expected.

# 1. Data Cleaning (SQL)

Data cleaning is performed via:

- sql/create_tables.sql giving the table `satellites` inside `data/processed/satellites.db` 
- sql/data_cleaning.sql giving the cleaned SQL table: `satellites_clean` inside `data/processed/satellites.db`.

Cleaning tasks include:

- removing corrupted records  
- deriving orbital features  
- removing ultra-rare object-types.

# 2. Exploratory Data Analysis (SQL + Python)

The notebook `01_sql_exploration.ipynb` includes:

- distribution of object types, orbital classes and countries with more payloads
- altitude vs eccentricity analysis  
- inclination patterns  
- visualization with matplotlib and seaborn  


# 3. Machine Learning Models

The project contains two supervised tasks:

- **Classification:** Predict the orbital band  
- **Regression:** Predict the orbital period (minutes)


## 3.1 Classification — Predicting Orbital Band

Target: `orbital_band`

Shortcut features (`altitude_km`, `altitude_category`, `mean_motion`) are excluded to prevent leakage.

### Features

Numerical:
- inclination  
- eccentricity  
- launch_year_estimate  
- days_in_orbit_estimate  

Categorical:
- object_type  
- satellite_constellation  
- congestion_risk  
- orbit_lifetime_category  
- country  

### Models used:
- Logistic Regression  
- Random Forest Classifier  
- Gradient Boosting Classifier  

Tree-based models reach ~99% accuracy. Logistic Regression ~97–98%.

Saved as:
models/sklearn/classification_rf.joblib
models/sklearn/classification_gb.joblib
models/sklearn/classification_logreg.joblib


## 3.2 Regression — Predicting Orbital Period

Target: `period_minutes`

Features same as in classification, plus `orbital_band`.

Shortcut features like `mean_motion` are excluded (they would trivially determine the period).

Models trained:
- Linear Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  

Metrics reported:
- MAE  
- RMSE  
- R²  

Saved as:
models/sklearn/regression_linreg.joblib
models/sklearn/regression_rf_reg.joblib
models/sklearn/regression_gb_reg.joblib


# 4. Neural Network Classifier (Keras)

A feed-forward neural network is implemented to classify `orbital_band`.

Architecture: Input → Dense(64, relu) → Dense(32, relu) → Dense(num_classes, softmax)

Preprocessing uses a scikit-learn ColumnTransformer:

- StandardScaler for numeric features  
- OneHotEncoder for categorical features  

Saved model artifacts:
models/keras/keras_classification_model.h5
models/keras/preprocessor.joblib
models/keras/class_labels.npy

# 5. Training the Models

All models can be retrained via `.py` scripts (no need for notebooks).


## Train classification models
python src/train_classification.py

## Train regression models
python src/train_regression.py

## Train Keras neural network
python src/train_keras.py

After training, new model files will appear in `models/`.


# 6. Using the Trained Models (Inference)

Utility functions are available in:
src/inference.py

## 6.1 Example: Use a scikit-learn classification model

```python
from src.inference import load_sklearn_model
import pandas as pd

model = load_sklearn_model("classification_rf")

X_new = pd.DataFrame([{
    "inclination": 98.7,
    "eccentricity": 0.0012,
    "launch_year_estimate": 2018,
    "days_in_orbit_estimate": 550,
    "object_type": "Payload",
    "satellite_constellation": "Starlink",
    "congestion_risk": "High",
    "orbit_lifetime_category": "5-15 years",
    "country": "USA"
}])

pred = model.predict(X_new)
print(pred[0])
```

## 6.2 Example: Use the Keras model

```python
from src.inference import load_keras_model, predict_keras
import pandas as pd

model, preprocessor, class_labels = load_keras_model()

X_new = pd.DataFrame([{
    "inclination": 97.5,
    "eccentricity": 0.0009,
    "launch_year_estimate": 2020,
    "days_in_orbit_estimate": 400,
    "object_type": "Payload",
    "satellite_constellation": "None",
    "congestion_risk": "Medium",
    "orbit_lifetime_category": "5-15 years",
    "country": "Russia"
}])

pred = predict_keras(model, preprocessor, class_labels, X_new)
print(pred[0])

```

# 7. Reproducibility
- All scikit-learn models use <code> random_state = 42 </code>
- Keras uses
```python
numpy.random.seed(42)
random.seed(42)
tf.random.set_seed(42)
```

# 8. Installation/How to run

This section explains how to set up the environment, run the notebooks, train the models, and perform inference using the trained artifacts.

---

## 8.1 Clone the Repository

```
git clone https://github.com/<your-username>/satellite-orbit-analyticsgit

cd satellite-orbit-analytics
```

## 8.2 Install environment

**Option A - pip**
```
pip install -r requirements.txt
pip install tensorflow==2.15 #CPU version
```

**Option B - conda**
```
conda create -n satelliteenv python=3.10
conda activate satelliteenv
pip install -r requirements.txt
pip install tensorflow==2.15
```

## 8.3 (Optional) Download the raw CSV

If you only want to **train and use the models**, you can skip this step:  
the cleaned database `data/processed/satellites.db` is already included.

If you want to **rebuild the database from the raw CSV**, you can:
1. Download the original `satellite_orbital_catalog.csv` from the data source.
2. Place it under `data/raw/`.
3. Use the SQL scripts and the notebook `notebooks/01_sql_exploration.ipynb` to recreate `satellites.db`.

## 8.5 **Database**

`data/processed/satellites.db` is already provided, so you can directly train the models.

If you prefer to rebuild it from scratch, see the optional step above.

## 8.6 Run the Notebooks (Optional)

If you want to explore the analysis or re-run experiments: 

```jupyter notebbok```

Then open:

- `01_sql_exploration.ipynb`  
- `02_classification_models.ipynb`  
- `03_regression_models.ipynb`  
- `04_keras_neural_network.ipynb`

These notebooks reproduce data analysis, model training, and evaluations.

## 8.7 Train the Models (Command Line)

All models can be trained without opening Jupyter. See Secs. 5 and 6

## 8.8 Troubleshooting

7. Troubleshooting

- TensorFlow not found: 
Install CPU version manually:

```pip install tensorflow==2.15```

- Database not found:
Ensure data/processed/satellites.db exists. Re-run SQL scripts if necessary.

```ImportError for src modules:
Run Python from project root:

python src/train_classification.py```

- Keras training starts at high accuracy:
Restart kernel or call:

```import tensorflow.keras.backend as K
K.clear_session()```

# 9. Future Work

- Orbital decay prediction
- Time-series modelling (LSTM/GRU with TLE snapshots)
- Collision risk estimation
- Streamlit/Power BI interactive dashboard
- Synthetic satellite population modelling
- Update graphics according to monthly updates of the raw dataset

# License

MIT license



