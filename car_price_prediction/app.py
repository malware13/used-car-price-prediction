import re
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

CURRENT_YEAR = 2025

def simplify_transmission(t):
    t = str(t).lower()
    if "manual" in t or "m/t" in t or re.search(r"\dm/t", t):
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
    """
    Replicates every transformation in train_model.py so the
    DataFrame columns match exactly what the pipeline was trained on.
    """
    car_age          = CURRENT_YEAR - model_year
    mileage_per_year = milage / (car_age + 1)
    mileage_sq       = milage ** 2
    age_sq           = car_age ** 2
    age_x_mileage    = car_age * milage

    fuel_remap = {"–": "unknown", "not supported": "unknown",
                  "nan": "unknown", "": "unknown"}
    fuel_type_clean      = fuel_remap.get(fuel_type, fuel_type)
    transmission_clean   = simplify_transmission(transmission)
    ext_col_clean        = simplify_ext_col(ext_col)

    # Engine features
    engine_hp        = extract_engine_hp(engine)
    engine_cylinders = extract_cylinders(engine)

    return pd.DataFrame([{
        # ── categorical ──
        "brand":              brand,
        "model":              model_name,
        "fuel_type":          fuel_type_clean,
        "transmission_clean": transmission_clean,
        "ext_col_clean":      ext_col_clean,
        # ── numeric ──
        "model_year":         model_year,
        "car_age":            car_age,
        "milage":             milage,
        "accident":           int(accident),
        "engine_hp":          engine_hp,
        "engine_cylinders":   engine_cylinders,
        "mileage_per_year":   mileage_per_year,
        "mileage_sq":         mileage_sq,
        "age_sq":             age_sq,
        "age_x_mileage":      age_x_mileage,
    }])

df = pd.read_csv("used_cars.csv")
df.columns = df.columns.str.strip().str.lower()

for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype(str).str.strip().str.lower()

df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")
df["car_age"]    = CURRENT_YEAR - df["model_year"]

df["milage"] = (
    df["milage"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.extract(r"(\d+)")
)
df["milage"] = pd.to_numeric(df["milage"], errors="coerce")

if "accident" in df.columns:
    df["accident"] = df["accident"].fillna("none").astype(str).str.lower()
    df["accident"] = df["accident"].apply(
        lambda x: 0 if any(w in x for w in ["none", "no", "clean"]) else 1
    )
else:
    df["accident"] = 0

fuel_remap = {"–": "unknown", "not supported": "unknown",
              "nan": "unknown", "": "unknown"}
df["fuel_type"] = df["fuel_type"].replace(fuel_remap)

model = joblib.load("rf_model.pkl")

@app.route("/")
def index():
    brands = sorted(df["brand"].dropna().unique())
    return render_template("index.html", brands=brands)


@app.route("/get_models/<brand>")
def get_models(brand):
    brand = brand.strip().lower()
    models = (
        df[df["brand"] == brand]["model"]
        .dropna().unique().tolist()
    )
    return jsonify(sorted(models))


@app.route("/get_years/<brand>/<model>")
def get_years(brand, model):
    brand = brand.strip().lower()
    model = model.strip().lower()
    years = (
        df[(df["brand"] == brand) & (df["model"] == model)]["model_year"]
        .dropna().astype(int).unique().tolist()
    )
    return jsonify(sorted(years))


@app.route("/get_car_data/<brand>/<model>/<year>")
def get_car_data(brand, model, year):
    try:
        year = int(year)
    except ValueError:
        return jsonify({"error": "Invalid year"})

    car = df[
        (df["brand"] == brand.strip().lower()) &
        (df["model"] == model.strip().lower()) &
        (df["model_year"] == year)
    ]

    if car.empty:
        return jsonify({"error": "Car not found"})

    return jsonify(car.iloc[0].to_dict())


@app.route("/predict", methods=["POST"])
def predict():
    # ── required fields ──
    brand      = request.form.get("brand",      "").strip().lower()
    model_name = request.form.get("model",      "").strip().lower()
    year_str   = request.form.get("year",       "")

    if not brand or not model_name or not year_str:
        return jsonify({"error": "Please select Brand, Model and Year"})

    try:
        year = int(year_str)
    except ValueError:
        return jsonify({"error": "Invalid year"})

    car_row = df[
        (df["brand"] == brand) &
        (df["model"] == model_name) &
        (df["model_year"] == year)
    ]

    if car_row.empty:
        return jsonify({"error": "Car not found in dataset"})

    car = car_row.iloc[0]

    fuel_type    = request.form.get("fuel_type",    str(car.get("fuel_type", "gasoline"))).strip().lower()
    transmission = request.form.get("transmission", str(car.get("transmission", "automatic"))).strip().lower()
    ext_col      = request.form.get("ext_col",      str(car.get("ext_col", "other"))).strip().lower()
    accident     = request.form.get("accident",     str(car.get("accident", 0)))
    engine       = str(car.get("engine", ""))

    try:
        milage = float(car.get("milage", 0))
    except (ValueError, TypeError):
        milage = 0.0

    X = build_features(
        brand        = brand,
        model_name   = model_name,
        model_year   = year,
        milage       = milage,
        fuel_type    = fuel_type,
        transmission = transmission,
        ext_col      = ext_col,
        accident     = accident,
        engine       = engine,
    )

    try:
        log_pred   = model.predict(X)[0]
        prediction = np.expm1(log_pred)
    except Exception as e:
        return jsonify({"error": str(e)})

    return jsonify({"predicted_price": round(float(prediction), 2)})

if __name__ == "__main__":
    app.run(debug=True)