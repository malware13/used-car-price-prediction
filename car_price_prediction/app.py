import re
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

CURRENT_YEAR = 2025

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Helper Functions
# ----------------------------

def simplify_transmission(t):
    t = str(t).lower()
    if "manual" in t or "m/t" in t:
        return "manual"
    if "cvt" in t:
        return "cvt"
    return "automatic"


def simplify_ext_col(color):
    top = ["black", "white", "gray", "silver", "blue", "red", "brown"]
    return color if color in top else "other"


def extract_engine_hp(engine_str):
    m = re.search(r"(\d+\.?\d*)\s*hp", str(engine_str), re.IGNORECASE)
    return float(m.group(1)) if m else np.nan


def extract_cylinders(engine_str):
    text = str(engine_str).lower()
    m = re.search(r"[vVlLiI](\d+)|(\d+)\s*cylinder", text)
    if m:
        return float(m.group(1) or m.group(2))
    return np.nan


def build_features(brand, model_name, model_year, milage, fuel_type,
                   transmission, ext_col, accident, engine=""):

    car_age = CURRENT_YEAR - model_year
    mileage_per_year = milage / (car_age + 1)
    mileage_sq = milage ** 2
    age_sq = car_age ** 2
    age_x_mileage = car_age * milage

    fuel_remap = {"–": "unknown", "not supported": "unknown",
                  "nan": "unknown", "": "unknown"}
    fuel_type_clean = fuel_remap.get(fuel_type, fuel_type)

    transmission_clean = simplify_transmission(transmission)
    ext_col_clean = simplify_ext_col(ext_col)

    engine_hp = extract_engine_hp(engine)
    engine_cylinders = extract_cylinders(engine)

    return pd.DataFrame([{
        "brand": brand,
        "model": model_name,
        "fuel_type": fuel_type_clean,
        "transmission_clean": transmission_clean,
        "ext_col_clean": ext_col_clean,

        "model_year": model_year,
        "car_age": car_age,
        "milage": milage,
        "accident": int(accident),
        "engine_hp": engine_hp,
        "engine_cylinders": engine_cylinders,
        "mileage_per_year": mileage_per_year,
        "mileage_sq": mileage_sq,
        "age_sq": age_sq,
        "age_x_mileage": age_x_mileage,
    }])

# ----------------------------
# Load Data
# ----------------------------

df = pd.read_csv(os.path.join(BASE_DIR, "used_cars.csv"))
df.columns = df.columns.str.strip().str.lower()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")
df["milage"] = (
    df["milage"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.extract(r"(\d+)")
)
df["milage"] = pd.to_numeric(df["milage"], errors="coerce")

df["accident"] = df["accident"].fillna("none").astype(str).str.lower()
df["accident"] = df["accident"].apply(
    lambda x: 0 if any(w in x for w in ["none", "no", "clean"]) else 1
)

model = joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("🚗 Used Car Price Prediction")

brand = st.selectbox(
    "Select Brand",
    sorted(df["brand"].dropna().unique())
)

models = sorted(df[df["brand"] == brand]["model"].unique())
model_name = st.selectbox("Select Model", models)

years = sorted(
    df[(df["brand"] == brand) & (df["model"] == model_name)]["model_year"]
    .dropna().astype(int).unique()
)

year = st.selectbox("Select Model Year", years)

car_row = df[
    (df["brand"] == brand) &
    (df["model"] == model_name) &
    (df["model_year"] == year)
]

if not car_row.empty:

    car = car_row.iloc[0]

    milage = float(car.get("milage", 0))
    fuel_type = car.get("fuel_type", "gasoline")
    transmission = car.get("transmission", "automatic")
    ext_col = car.get("ext_col", "other")
    accident = car.get("accident", 0)
    engine = car.get("engine", "")

    st.write("### Car Details")
    st.write(car)

    if st.button("Predict Price"):

        X = build_features(
            brand=brand,
            model_name=model_name,
            model_year=year,
            milage=milage,
            fuel_type=fuel_type,
            transmission=transmission,
            ext_col=ext_col,
            accident=accident,
            engine=engine
        )

        log_pred = model.predict(X)[0]
        prediction = np.expm1(log_pred)

        st.success(f"Estimated Price: ${prediction:,.2f}")
