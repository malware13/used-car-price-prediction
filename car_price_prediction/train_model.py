import re
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════
print("=" * 55)
print("  USED CAR PRICE PREDICTOR — IMPROVED PIPELINE")
print("=" * 55)

df = pd.read_csv("used_cars.csv")
df.columns = df.columns.str.strip().str.lower()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

print(f"\n[1] Raw rows loaded       : {len(df):,}")

# ══════════════════════════════════════════════════════
# 2. CLEAN PRICE
# ══════════════════════════════════════════════════════
df["price"] = (
    df["price"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
)
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# ══════════════════════════════════════════════════════
# 3. CLEAN MILEAGE
# ══════════════════════════════════════════════════════
df["milage"] = (
    df["milage"]
    .str.replace(",", "", regex=False)
    .str.extract(r"(\d+)")
)
df["milage"] = pd.to_numeric(df["milage"], errors="coerce")

# ══════════════════════════════════════════════════════
# 4. CLEAN MODEL YEAR
# ══════════════════════════════════════════════════════
df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")

# ══════════════════════════════════════════════════════
# 5. ACCIDENT  →  binary 0/1
# ══════════════════════════════════════════════════════
df["accident"] = df["accident"].apply(
    lambda x: 0 if "none" in str(x).lower() else 1
)

# ══════════════════════════════════════════════════════
# 6. CLEAN TITLE  →  binary 0/1
#    NEW: clean_title column added
# ══════════════════════════════════════════════════════
df["clean_title"] = df["clean_title"].apply(
    lambda x: 1 if "yes" in str(x).lower() else 0
)

# ══════════════════════════════════════════════════════
# 7. FUEL TYPE normalisation
# ══════════════════════════════════════════════════════
fuel_remap = {"–": "unknown", "not supported": "unknown",
              "nan": "unknown", "": "unknown"}
df["fuel_type"] = df["fuel_type"].replace(fuel_remap)

# ══════════════════════════════════════════════════════
# 8. EXTRACT ENGINE HP
# ══════════════════════════════════════════════════════
df["engine_hp"] = (
    df["engine"]
    .astype(str)
    .str.extract(r"(\d+\.?\d*)\s*hp", flags=re.IGNORECASE)
    .astype(float)
)

# ══════════════════════════════════════════════════════
# 9. EXTRACT ENGINE CYLINDERS
# ══════════════════════════════════════════════════════
def extract_cylinders(text):
    text = str(text).lower()
    m = re.search(r"[vVlLiI](\d+)|(\d+)\s*cylinder", text)
    if m:
        return float(m.group(1) or m.group(2))
    return np.nan

df["engine_cylinders"] = df["engine"].apply(extract_cylinders)

# ══════════════════════════════════════════════════════
# 10. EXTRACT ENGINE DISPLACEMENT (litres)
#     NEW: e.g. "3.7l" → 3.7
# ══════════════════════════════════════════════════════
df["engine_displacement"] = (
    df["engine"]
    .astype(str)
    .str.extract(r"(\d+\.?\d*)\s*l\b", flags=re.IGNORECASE)
    .astype(float)
)

# ══════════════════════════════════════════════════════
# 11. SIMPLIFY TRANSMISSION
# ══════════════════════════════════════════════════════
def simplify_transmission(t):
    t = str(t).lower()
    if "manual" in t or "m/t" in t or re.search(r"\dm/t", t):
        return "manual"
    if "cvt" in t:
        return "cvt"
    return "automatic"

df["transmission_clean"] = df["transmission"].apply(simplify_transmission)

# ══════════════════════════════════════════════════════
# 12. EXTERIOR COLOR  →  top 7 + "other"
# ══════════════════════════════════════════════════════
top_colors = ["black", "white", "gray", "silver", "blue", "red", "brown"]
df["ext_col_clean"] = df["ext_col"].apply(
    lambda x: x if x in top_colors else "other"
)

# ══════════════════════════════════════════════════════
# 13. DROP INCOMPLETE ROWS + OUTLIER CLIP
# ══════════════════════════════════════════════════════
df = df.dropna(subset=["price", "model_year", "milage"])
df = df[df["price"] > df["price"].quantile(0.005)]
df = df[df["price"] < df["price"].quantile(0.995)]

print(f"[2] Rows after cleaning   : {len(df):,}")

# ══════════════════════════════════════════════════════
# 14. FEATURE ENGINEERING  (IMPROVED)
# ══════════════════════════════════════════════════════
CURRENT_YEAR = 2025

df["car_age"]              = CURRENT_YEAR - df["model_year"]
df["mileage_per_year"]     = df["milage"] / (df["car_age"] + 1)
df["mileage_sq"]           = df["milage"] ** 2
df["age_sq"]               = df["car_age"] ** 2
df["age_x_mileage"]        = df["car_age"] * df["milage"]
# NEW engineered features
df["hp_per_cylinder"]      = df["engine_hp"] / (df["engine_cylinders"] + 1)
df["hp_per_litre"]         = df["engine_hp"] / (df["engine_displacement"] + 0.001)
df["value_density"]        = df["engine_hp"] / (df["car_age"] + 1)

print(f"[3] Features engineered   : car_age, mileage_per_year, hp_per_cylinder,")
print(f"                            hp_per_litre, value_density, age_x_mileage +")

# ══════════════════════════════════════════════════════
# 15. TARGET  —  log1p to reduce price skew
# ══════════════════════════════════════════════════════
y = np.log1p(df["price"])

# ══════════════════════════════════════════════════════
# 16. FEATURE LISTS
# ══════════════════════════════════════════════════════
categorical = [
    "brand",
    "model",
    "fuel_type",
    "transmission_clean",
    "ext_col_clean",
]

numeric = [
    "model_year",
    "car_age",
    "milage",
    "accident",
    "clean_title",           # NEW
    "engine_hp",
    "engine_cylinders",
    "engine_displacement",   # NEW
    "mileage_per_year",
    "mileage_sq",
    "age_sq",
    "age_x_mileage",
    "hp_per_cylinder",       # NEW
    "hp_per_litre",          # NEW
    "value_density",         # NEW
]

features = categorical + numeric
X = df[features].copy()

print(f"[4] Total features        : {len(features)} ({len(categorical)} categorical, {len(numeric)} numeric)")

# ══════════════════════════════════════════════════════
# 17. PREPROCESSOR
#     HistGradientBoosting natively handles NaN,
#     but we still encode categoricals with OrdinalEncoder
#     (faster than OHE and works with HGBR)
# ══════════════════════════════════════════════════════
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1)),
])

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

preprocessor = ColumnTransformer([
    ("cat", categorical_transformer, categorical),
    ("num", numeric_transformer,     numeric),
], remainder="drop")

# ══════════════════════════════════════════════════════
# 18. MODEL  —  HistGradientBoostingRegressor
#     Equivalent to XGBoost; native NaN support,
#     faster on large datasets, built into sklearn.
#     Tuned hyperparameters via domain knowledge + CV.
# ══════════════════════════════════════════════════════
model = HistGradientBoostingRegressor(
    max_iter          = 800,        # equivalent to n_estimators
    learning_rate     = 0.03,       # conservative → better generalisation
    max_depth         = 7,
    min_samples_leaf  = 20,         # regularisation: min 20 samples per leaf
    l2_regularization = 0.1,        # L2 penalty on leaf values
    max_features      = 0.8,        # column subsampling (like colsample_bytree)
    random_state      = 42,
    early_stopping    = True,       # NEW: stops when val score plateaus
    validation_fraction = 0.1,      # 10 % of training data used for early stop
    n_iter_no_change  = 30,         # patience = 30 rounds
    scoring           = "loss",
)

pipeline = Pipeline([
    ("prep",  preprocessor),
    ("model", model),
])

# ══════════════════════════════════════════════════════
# 19. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n[5] Train size            : {len(X_train):,} rows")
print(f"    Test  size            : {len(X_test):,} rows")

# ══════════════════════════════════════════════════════
# 20. TRAIN
# ══════════════════════════════════════════════════════
print("\n[6] Training model (early stopping enabled)...")
pipeline.fit(X_train, y_train)
n_iters = pipeline.named_steps["model"].n_iter_
print(f"    Stopped at iteration  : {n_iters}  (out of max 800)")
print("    Training complete ✓")

# ══════════════════════════════════════════════════════
# 21. EVALUATE ON TRAIN & TEST
# ══════════════════════════════════════════════════════
train_preds = pipeline.predict(X_train)
test_preds  = pipeline.predict(X_test)

# Back to real dollars
y_train_real  = np.expm1(y_train)
y_test_real   = np.expm1(y_test)
train_pred_real = np.expm1(train_preds)
test_pred_real  = np.expm1(test_preds)

train_r2   = r2_score(y_train, train_preds)
test_r2    = r2_score(y_test,  test_preds)
train_mae  = mean_absolute_error(y_train_real, train_pred_real)
test_mae   = mean_absolute_error(y_test_real,  test_pred_real)
train_rmse = np.sqrt(mean_squared_error(y_train_real, train_pred_real))
test_rmse  = np.sqrt(mean_squared_error(y_test_real,  test_pred_real))

# MAPE
test_mape = np.mean(np.abs((y_test_real - test_pred_real) / y_test_real)) * 100

print("\n" + "─" * 45)
print("  EVALUATION RESULTS")
print("─" * 45)
print(f"  {'Metric':<22} {'Train':>9}  {'Test':>9}")
print("─" * 45)
print(f"  {'R²':<22} {train_r2:>9.4f}  {test_r2:>9.4f}")
print(f"  {'MAE ($)':<22} {train_mae:>9,.0f}  {test_mae:>9,.0f}")
print(f"  {'RMSE ($)':<22} {train_rmse:>9,.0f}  {test_rmse:>9,.0f}")
print(f"  {'MAPE (%)':<22} {'—':>9}  {test_mape:>9.2f}")
print(f"  {'Overfit Gap (R²)':<22} {train_r2 - test_r2:>9.4f}  {'(< 0.05 healthy)':>9}")
print("─" * 45)

# ══════════════════════════════════════════════════════
# 22. CROSS VALIDATION  (5-fold stratified)
# ══════════════════════════════════════════════════════
print("\n[7] Running 5-fold cross-validation...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X, y, cv=kf, scoring="r2", n_jobs=-1)

print("\n" + "─" * 45)
print("  CROSS-VALIDATION (5-fold)")
print("─" * 45)
print(f"  Fold scores : {[round(s, 4) for s in cv_scores]}")
print(f"  Mean R²     : {cv_scores.mean():.4f}")
print(f"  Std dev     : {cv_scores.std():.4f}  (lower = more stable)")
print("─" * 45)

# ══════════════════════════════════════════════════════
# 23. LIVE PREDICTION TESTS
# ══════════════════════════════════════════════════════
print("\n[8] LIVE PREDICTION TESTS")
print("─" * 55)

test_cases = [
    {
        "label": "2021 Toyota Camry LE — Low mileage, no accident",
        "brand": "toyota", "model": "camry le", "model_year": 2021,
        "milage": 28000, "fuel_type": "gasoline", "engine_hp": 203.0,
        "engine_cylinders": 4.0, "engine_displacement": 2.5,
        "transmission_clean": "automatic", "ext_col_clean": "white",
        "accident": 0, "clean_title": 1,
    },
    {
        "label": "2018 Ford F-150 XLT — High mileage, 1 accident",
        "brand": "ford", "model": "f-150 xlt", "model_year": 2018,
        "milage": 89000, "fuel_type": "gasoline", "engine_hp": 290.0,
        "engine_cylinders": 6.0, "engine_displacement": 3.5,
        "transmission_clean": "automatic", "ext_col_clean": "silver",
        "accident": 1, "clean_title": 1,
    },
    {
        "label": "2015 Honda Civic EX — Very high mileage, no accident",
        "brand": "honda", "model": "civic ex", "model_year": 2015,
        "milage": 130000, "fuel_type": "gasoline", "engine_hp": 143.0,
        "engine_cylinders": 4.0, "engine_displacement": 1.8,
        "transmission_clean": "automatic", "ext_col_clean": "blue",
        "accident": 0, "clean_title": 1,
    },
    {
        "label": "2023 Tesla Model 3 — Nearly new, electric",
        "brand": "tesla", "model": "model 3", "model_year": 2023,
        "milage": 8000, "fuel_type": "electric", "engine_hp": 283.0,
        "engine_cylinders": np.nan, "engine_displacement": np.nan,
        "transmission_clean": "automatic", "ext_col_clean": "white",
        "accident": 0, "clean_title": 1,
    },
]

for tc in test_cases:
    label = tc.pop("label")
    # Build derived features
    car_age = CURRENT_YEAR - tc["model_year"]
    tc["car_age"]           = car_age
    tc["mileage_per_year"]  = tc["milage"] / (car_age + 1)
    tc["mileage_sq"]        = tc["milage"] ** 2
    tc["age_sq"]            = car_age ** 2
    tc["age_x_mileage"]     = car_age * tc["milage"]
    tc["hp_per_cylinder"]   = tc["engine_hp"] / (tc["engine_cylinders"] + 1) if not np.isnan(tc.get("engine_cylinders", np.nan)) else np.nan
    tc["hp_per_litre"]      = tc["engine_hp"] / (tc["engine_displacement"] + 0.001) if not np.isnan(tc.get("engine_displacement", np.nan)) else np.nan
    tc["value_density"]     = tc["engine_hp"] / (car_age + 1)

    row = pd.DataFrame([tc])[features]
    pred_log  = pipeline.predict(row)[0]
    pred_price = np.expm1(pred_log)
    print(f"  {label}")
    print(f"  → Predicted price: ${pred_price:,.0f}")
    print()

# ══════════════════════════════════════════════════════
# 24. SAVE MODEL
# ══════════════════════════════════════════════════════
joblib.dump(pipeline, "used_car_model.pkl")
joblib.dump(features,  "model_features.pkl")

print("─" * 55)
print(f"  ✅ Model saved  →  used_car_model.pkl")
print(f"  ✅ Feature list →  model_features.pkl")
print(f"  Final Test R²  →  {test_r2:.4f}")
print(f"  Final Test MAE →  ${test_mae:,.0f}")
print("─" * 55)