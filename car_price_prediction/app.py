import re
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

CURRENT_YEAR = 2025
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------
# Page Config
# ----------------------------

st.set_page_config(
    page_title="AutoVal — Car Price Estimator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------------------
# Custom CSS
# ----------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0a0a0f; color: #f0ede8; }
#MainMenu, footer, header { visibility: hidden; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.8rem;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #f0ede8 0%, #c8a96e 50%, #f0ede8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    font-size: 1.1rem;
    font-weight: 300;
    color: #7a7a8a;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #c8a96e;
    margin-bottom: 1rem;
}
.car-image-container {
    width: 100%;
    border-radius: 12px;
    overflow: hidden;
    background: #13131a;
    border: 1px solid #1e1e2e;
    height: 220px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.car-image-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 12px;
}
.stat-row {
    display: flex;
    gap: 0.8rem;
    flex-wrap: wrap;
    margin: 1rem 0;
}
.stat-pill {
    background: #1a1a24;
    border: 1px solid #2a2a3a;
    border-radius: 50px;
    padding: 0.4rem 1rem;
    font-size: 0.82rem;
    color: #b0adb8;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
}
.price-card {
    background: linear-gradient(135deg, #1a1508 0%, #13131a 100%);
    border: 1px solid #c8a96e55;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
}
.price-label {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #c8a96e;
    margin-bottom: 0.5rem;
}
.price-value {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    color: #f0ede8;
    letter-spacing: -0.02em;
    line-height: 1;
}
.price-note {
    font-size: 0.8rem;
    color: #5a5a6a;
    margin-top: 0.8rem;
}
.divider {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 1.5rem 0;
}
.stSelectbox > div > div {
    background: #13131a !important;
    border: 1px solid #2a2a3a !important;
    border-radius: 10px !important;
    color: #f0ede8 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #c8a96e, #a07840) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
</style>
""", unsafe_allow_html=True)


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
    fuel_remap = {"–": "unknown", "not supported": "unknown", "nan": "unknown", "": "unknown"}
    fuel_type_clean = fuel_remap.get(fuel_type, fuel_type)
    transmission_clean = simplify_transmission(transmission)
    ext_col_clean = simplify_ext_col(ext_col)
    engine_hp = extract_engine_hp(engine)
    engine_cylinders = extract_cylinders(engine)
    return pd.DataFrame([{
        "brand": brand, "model": model_name, "fuel_type": fuel_type_clean,
        "transmission_clean": transmission_clean, "ext_col_clean": ext_col_clean,
        "model_year": model_year, "car_age": car_age, "milage": milage,
        "accident": int(accident), "engine_hp": engine_hp,
        "engine_cylinders": engine_cylinders, "mileage_per_year": mileage_per_year,
        "mileage_sq": mileage_sq, "age_sq": age_sq, "age_x_mileage": age_x_mileage,
    }])

def get_car_image(brand):
    """
    Returns an Unsplash Source URL for a car image based on brand.
    Uses a fixed seed per brand so the image is consistent.
    No API key required.
    """
    seed = abs(hash(brand.lower())) % 1000
    query = f"{brand.lower()}-car-automobile"
    return f"https://source.unsplash.com/800x450/?{query}&sig={seed}"

def fuel_icon(fuel):
    return {"gasoline": "⛽", "electric": "⚡", "hybrid": "🔋", "diesel": "🛢️"}.get(str(fuel).lower(), "⛽")

def transmission_icon(t):
    return "🕹️" if "manual" in str(t).lower() else "🔄"


# ----------------------------
# Load Data & Model
# ----------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(os.path.join(BASE_DIR, "used_cars.csv"))
    data.columns = data.columns.str.strip().str.lower()
    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype(str).str.strip().str.lower()
    data["model_year"] = pd.to_numeric(data["model_year"], errors="coerce")
    data["milage"] = (
        data["milage"].astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+)")
    )
    data["milage"] = pd.to_numeric(data["milage"], errors="coerce")
    data["accident"] = data["accident"].fillna("none").astype(str).str.lower()
    data["accident"] = data["accident"].apply(
        lambda x: 0 if any(w in x for w in ["none", "no", "clean"]) else 1
    )
    return data

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))

df = load_data()
ml_model = load_model()

# ----------------------------
# UI
# ----------------------------

st.markdown('<div class="hero-title">AutoVal</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Intelligent Used Car Price Estimator</div>', unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.4], gap="large")

with col_left:
    st.markdown('<div class="section-label">🔍 Configure Your Search</div>', unsafe_allow_html=True)
    brand = st.selectbox("Brand", sorted(df["brand"].dropna().unique()))
    models_list = sorted(df[df["brand"] == brand]["model"].unique())
    model_name = st.selectbox("Model", models_list)
    years = sorted(
        df[(df["brand"] == brand) & (df["model"] == model_name)]["model_year"]
        .dropna().astype(int).unique()
    )
    year = st.selectbox("Model Year", years)
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    predict_clicked = st.button("✦ Estimate Price")

with col_right:
    car_row = df[
        (df["brand"] == brand) &
        (df["model"] == model_name) &
        (df["model_year"] == year)
    ]

    if not car_row.empty:
        car = car_row.iloc[0]
        milage = float(car.get("milage", 0) or 0)
        fuel_type = car.get("fuel_type", "gasoline")
        transmission = car.get("transmission", "automatic")
        ext_col = car.get("ext_col", "other")
        accident = car.get("accident", 0)
        engine = car.get("engine", "")
        engine_hp = extract_engine_hp(engine)
        engine_cyl = extract_cylinders(engine)

        st.markdown('<div class="section-label">📸 Vehicle Preview</div>', unsafe_allow_html=True)

        image_url = get_car_image(brand)
        st.markdown(f"""
        <div class="car-image-container">
            <img src="{image_url}" alt="{brand} car" />
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-top:1.2rem; font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:700; color:#f0ede8;">
            {year} {brand.title()} {model_name.title()}
        </div>""", unsafe_allow_html=True)

        pills = [f"📍 {int(milage):,} mi", f"{fuel_icon(fuel_type)} {fuel_type.title()}",
                 f"{transmission_icon(transmission)} {simplify_transmission(transmission).title()}"]
        if not np.isnan(engine_hp):
            pills.append(f"🔥 {int(engine_hp)} HP")
        if not np.isnan(engine_cyl):
            pills.append(f"🔩 {int(engine_cyl)}-cyl")
        pills.append(f"🎨 {ext_col.title()}")
        pills.append("⚠️ Accident History" if accident else "✅ Clean History")

        pills_html = "".join([f'<span class="stat-pill">{p}</span>' for p in pills])
        st.markdown(f'<div class="stat-row">{pills_html}</div>', unsafe_allow_html=True)

        if predict_clicked:
            X = build_features(
                brand=brand, model_name=model_name, model_year=year,
                milage=milage, fuel_type=fuel_type, transmission=transmission,
                ext_col=ext_col, accident=accident, engine=engine
            )
            log_pred = ml_model.predict(X)[0]
            prediction = np.expm1(log_pred)
            st.markdown(f"""
            <div class="price-card">
                <div class="price-label">Estimated Market Value</div>
                <div class="price-value">${prediction:,.0f}</div>
                <div class="price-note">Based on {len(df):,} comparable vehicles · AI-powered estimate</div>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="height:300px; display:flex; align-items:center; justify-content:center; color:#3a3a4a; font-size:1rem;">
            Select a brand, model, and year to get started
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div style="margin-top:4rem; padding-top:1.5rem; border-top:1px solid #1e1e2e; text-align:center; color:#3a3a4a; font-size:0.78rem; letter-spacing:0.05em;">
    AUTOVAL · Powered by Machine Learning · Estimates are for reference only
</div>""", unsafe_allow_html=True)
