from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scipy import sparse


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw" / "city_day.csv"
OUT = ROOT / "data" / "processed"

OUT.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(RAW)


# --------------------------------------------------
# Date feature engineering
# --------------------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df["YEAR"] = df["Date"].dt.year
df["MONTH"] = df["Date"].dt.month
df["DAY"] = df["Date"].dt.day
df["DAY_OF_WEEK"] = df["Date"].dt.dayofweek


# --------------------------------------------------
# Remove rows without AQI
# --------------------------------------------------

work = df.dropna(
    subset=["AQI"]
).copy()


# --------------------------------------------------
# Target
# AQI >= 200 = high AQI
# --------------------------------------------------

work["AQI_Category"] = (
    work["AQI"] >= 200
).astype(int)


# --------------------------------------------------
# Features
# --------------------------------------------------

features = [
    c for c in [
        "City",
        "PM2.5",
        "PM10",
        "NO",
        "NO2",
        "NOx",
        "NH3",
        "CO",
        "SO2",
        "O3",
        "Benzene",
        "Toluene",
        "Xylene",
        "YEAR",
        "MONTH",
        "DAY",
        "DAY_OF_WEEK"
    ]
    if c in work.columns
]


X = work[features].copy()
y = work["AQI_Category"].copy()


# --------------------------------------------------
# Train-test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# --------------------------------------------------
# Identify columns
# --------------------------------------------------

cat_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

num_cols = [
    c for c in features
    if c not in cat_cols
]


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

pre = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                (
                    "imp",
                    SimpleImputer(
                        strategy="median"
                    )
                ),
                (
                    "scale",
                    StandardScaler()
                )
            ]),
            num_cols
        ),
        (
            "cat",
            Pipeline([
                (
                    "imp",
                    SimpleImputer(
                        strategy="most_frequent"
                    )
                ),
                (
                    "ohe",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]),
            cat_cols
        )
    ]
)


# --------------------------------------------------
# Transform
# --------------------------------------------------

Xtr = pre.fit_transform(X_train)
Xte = pre.transform(X_test)


# --------------------------------------------------
# Ensure sparse format
# --------------------------------------------------

if not sparse.issparse(Xtr):
    Xtr = sparse.csr_matrix(Xtr)

if not sparse.issparse(Xte):
    Xte = sparse.csr_matrix(Xte)


# --------------------------------------------------
# Save processed data
# --------------------------------------------------

sparse.save_npz(
    OUT / "X_train_final.npz",
    Xtr
)

sparse.save_npz(
    OUT / "X_test_final.npz",
    Xte
)

np.save(
    OUT / "y_train.npy",
    y_train.to_numpy()
)

np.save(
    OUT / "y_test.npy",
    y_test.to_numpy()
)


print(
    f"[SUCCESS] Preprocessing complete: "
    f"train={Xtr.shape}, test={Xte.shape}"
)