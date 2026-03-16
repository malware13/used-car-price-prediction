import re
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

CURRENT_YEAR = 2025
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GITHUB_RAW = "https://raw.githubusercontent.com/malware13/used-car-price-prediction/main/car_price_prediction/static/images"

ALL_IMAGES = [
    "bugatti.jpg", "ford.jpg", "hummer.jpg", "lambo.webp",
    "lykan.jpg", "mitsubishi.webp", "mustang.jpg", "nissan.webp",
]

BRAND_IMAGE_MAP = {
    "bugatti":     "bugatti.jpg",
    "ford":        "ford.jpg",
    "hummer":      "hummer.jpg",
    "lamborghini": "lambo.webp",
    "lykan":       "lykan.jpg",
    "mitsubishi":  "mitsubishi.webp",
    "mustang":     "mustang.jpg",
    "nissan":      "nissan.webp",
}

def get_car_image(brand):
    brand_lower = str(brand).lower()
    for key, img in BRAND_IMAGE_MAP.items():
        if key in brand_lower:
            return f"{GITHUB_RAW}/{img}"
    img = ALL_IMAGES[abs(hash(brand_lower)) % len(ALL_IMAGES)]
    return f"{GITHUB_RAW}/{img}"


st.set_page_config(
    page_title="Used Car Predictor",
    page_icon="🏁",
    layout="centered",
    initial_sidebar_state="collapsed"
)

BG_IMAGE = f"{GITHUB_RAW}/1.webp"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700&family=Syne:wght@400;500;600;700&display=swap');

:root {{
  --bg: #080a0f;
  --surface: #0e1118;
  --surface2: #141820;
  --surface3: #1c2030;
  --gold: #c9a84c;
  --gold-light: #e8c86a;
  --gold-dim: rgba(201,168,76,0.10);
  --teal: #2dd4bf;
  --teal-dim: rgba(45,212,191,0.10);
  --text: #eef0f5;
  --text-2: #7c8496;
  --text-3: #3e4454;
  --border: rgba(255,255,255,0.06);
  --border-gold: rgba(201,168,76,0.30);
  --r: 10px;
  --r-lg: 18px;
}}

html, body, [class*="css"] {{
  font-family: 'Syne', sans-serif;
  color: var(--text);
}}

/* Background using 1.webp */
.stApp {{
  background-image:
    radial-gradient(ellipse 70% 40% at 50% 0%, rgba(201,168,76,0.07) 0%, transparent 65%),
    radial-gradient(ellipse 40% 30% at 85% 80%, rgba(45,212,191,0.04) 0%, transparent 60%),
    linear-gradient(170deg, rgba(8,10,15,0.55) 0%, rgba(8,10,15,0.35) 50%, rgba(8,10,15,0.6) 100%),
    url("{BG_IMAGE}") !important;
  background-size: cover !important;
  background-position: center top !important;
  background-repeat: no-repeat !important;
  background-attachment: scroll !important;
  background-color: var(--bg) !important;
}}

#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding-top: 2rem !important; padding-bottom: 2rem !important; }}

/* ── Card ── */
.card {{
  width: 100%;
  max-width: 480px;
  margin: 0 auto;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  box-shadow:
    0 0 0 1px rgba(255,255,255,0.03),
    0 32px 80px rgba(0,0,0,0.7),
    0 0 60px rgba(201,168,76,0.06);
  overflow: hidden;
  animation: riseIn 0.7s cubic-bezier(0.22,1,0.36,1) both;
}}

/* ── Card Header ── */
.card-header {{
  padding: 30px 32px 26px;
  background: linear-gradient(150deg, #11151e 0%, #0c0f17 100%);
  border-bottom: 1px solid var(--border);
  position: relative;
  overflow: hidden;
}}
.card-header::before {{
  content: "";
  position: absolute; inset: 0;
  background-image:
    linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
  background-size: 32px 32px;
  mask-image: radial-gradient(ellipse at center, black 30%, transparent 80%);
  pointer-events: none;
}}
.card-header::after {{
  content: "";
  position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--gold) 40%, var(--teal) 70%, transparent);
  opacity: 0.5;
}}
.header-top {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }}
.pill {{
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 10px; font-weight: 600; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--gold);
  background: var(--gold-dim); border: 1px solid var(--border-gold);
  border-radius: 100px; padding: 5px 13px;
}}
.pill-dot {{
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--gold); display: inline-block;
  animation: blink 2.2s ease-in-out infinite;
}}
.header-right {{ display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }}
.model-tag {{ font-size: 10px; font-weight: 600; letter-spacing: 0.10em; text-transform: uppercase; color: var(--text-3); }}
.accuracy-badge {{
  background: rgba(45,212,191,0.08); border: 1px solid rgba(45,212,191,0.25);
  border-radius: 8px; padding: 5px 10px; text-align: center;
}}
.accuracy-num {{ font-family: 'Barlow Condensed', sans-serif; font-size: 18px; font-weight: 700; color: var(--teal); letter-spacing: 0.04em; }}
.card-title {{ font-family: 'Barlow Condensed', sans-serif; font-size: 40px; font-weight: 700; letter-spacing: 0.05em; line-height: 1; color: var(--text); margin-bottom: 6px; }}
.card-title em {{ font-style: normal; color: var(--gold); }}
.card-sub {{ font-size: 12.5px; color: var(--text-2); font-weight: 400; margin-bottom: 20px; }}
.steps {{ display: flex; gap: 6px; }}
.step {{ flex: 1; height: 3px; border-radius: 10px; background: var(--surface3); }}
.step-active {{ background: var(--gold); }}
.step-done   {{ background: var(--teal); }}

.card-body {{ padding: 28px 32px 32px; }}
.field {{ margin-bottom: 16px; }}
.field-label {{
  display: flex; align-items: center; gap: 8px;
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.13em;
  text-transform: uppercase; color: #a8b0c0; margin-bottom: 7px;
}}
.field-num {{
  width: 18px; height: 18px; border-radius: 50%;
  background: var(--surface3); border: 1px solid rgba(255,255,255,0.15);
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 9px; color: #a8b0c0; font-weight: 700;
}}
.field-num-filled {{ background: var(--gold-dim); border-color: var(--border-gold); color: var(--gold); }}
.divider-line {{ border: none; border-top: 1px solid var(--border); margin: 22px 0; }}

.car-image-wrap {{
  position: relative; width: 100%; height: 180px;
  border-radius: var(--r); overflow: hidden;
  margin-bottom: 18px; border: 1px solid var(--border);
}}
.car-image-wrap img {{ width: 100%; height: 100%; object-fit: cover; filter: brightness(0.85); }}
.car-image-overlay {{
  position: absolute; bottom: 0; left: 0; right: 0; padding: 10px 14px;
  background: linear-gradient(0deg, rgba(8,10,15,0.85) 0%, transparent 100%);
}}
.car-image-overlay span {{
  font-family: 'Barlow Condensed', sans-serif; font-size: 15px;
  font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: var(--text);
}}

.specs-box {{ background: var(--surface2); border: 1px solid var(--border); border-radius: var(--r); padding: 16px 18px; margin-bottom: 18px; }}
.specs-title {{ font-size: 9.5px; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase; color: var(--text-3); margin-bottom: 12px; }}
.spec-row {{ display: flex; align-items: center; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid var(--border); font-size: 13px; }}
.spec-row:last-child {{ border-bottom: none; }}
.spec-key {{ color: #a8b0c0; font-weight: 500; }}
.spec-val {{ color: var(--text); font-weight: 600; }}
.spec-val-green {{ color: #3ecf8e; font-weight: 600; }}
.spec-val-red {{ color: #f87171; font-weight: 600; }}

.refine-title {{
  font-size: 9.5px; font-weight: 700; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--text-3); margin-bottom: 14px;
  display: flex; align-items: center; gap: 10px;
}}
.refine-title::before, .refine-title::after {{ content: ""; flex: 1; height: 1px; background: var(--border); }}

.result-panel {{ margin-top: 18px; border-radius: var(--r); overflow: hidden; border: 1px solid var(--border-gold); animation: fadeSlide 0.45s cubic-bezier(0.22,1,0.36,1) both; }}
.result-top {{ background: linear-gradient(135deg, rgba(201,168,76,0.08) 0%, rgba(45,212,191,0.04) 100%); padding: 22px 24px 18px; text-align: center; border-bottom: 1px solid var(--border); }}
.result-label {{ font-size: 10px; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; color: var(--text-2); margin-bottom: 10px; }}
.result-price {{ font-family: 'Barlow Condensed', sans-serif; font-size: 56px; font-weight: 700; letter-spacing: 0.03em; color: var(--gold-light); line-height: 1; text-shadow: 0 0 40px rgba(201,168,76,0.45); }}
.result-range {{ font-size: 12px; color: var(--text-3); margin-top: 7px; }}
.result-bottom {{ background: var(--surface2); padding: 14px 24px; display: flex; align-items: center; gap: 8px; }}
.result-dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--teal); flex-shrink: 0; box-shadow: 0 0 8px rgba(45,212,191,0.6); display: inline-block; }}
.result-note {{ font-size: 11.5px; color: var(--text-2); line-height: 1.4; }}

.card-footer {{ padding: 14px 32px; background: var(--surface); border-top: 1px solid var(--border); display: flex; align-items: center; justify-content: center; gap: 8px; }}
.footer-dot {{ width: 5px; height: 5px; border-radius: 50%; background: var(--teal); opacity: 0.6; display: inline-block; }}
.footer-text {{ font-size: 10.5px; color: var(--text-3); letter-spacing: 0.06em; }}

.stSelectbox label {{ display: none !important; }}
.stSelectbox > div > div {{
  background: var(--surface2) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r) !important; color: var(--text) !important;
  font-family: 'Syne', sans-serif !important; font-size: 14px !important;
}}
.stSelectbox > div > div:focus-within {{ border-color: var(--border-gold) !important; box-shadow: 0 0 0 3px var(--gold-dim) !important; }}
.stButton > button {{
  width: 100% !important; padding: 16px !important;
  background: linear-gradient(135deg, var(--gold) 0%, #b8922e 100%) !important;
  color: #07090d !important; border: none !important; border-radius: var(--r) !important;
  font-family: 'Barlow Condensed', sans-serif !important; font-size: 20px !important;
  font-weight: 700 !important; letter-spacing: 0.12em !important; text-transform: uppercase !important;
  box-shadow: 0 6px 24px rgba(201,168,76,0.28) !important;
}}
.stButton > button:hover {{ filter: brightness(1.08) !important; }}

@keyframes riseIn {{
  from {{ opacity: 0; transform: translateY(32px) scale(0.98); }}
  to   {{ opacity: 1; transform: translateY(0) scale(1); }}
}}
@keyframes fadeSlide {{
  from {{ opacity: 0; transform: translateY(8px); }}
  to   {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes blink {{
  0%, 100% {{ opacity: 1; transform: scale(1); }}
  50%       {{ opacity: 0.3; transform: scale(0.7); }}
}}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Helper Functions
# ----------------------------

def simplify_transmission(t):
    t = str(t).lower()
    if "manual" in t or "m/t" in t: return "manual"
    if "cvt" in t: return "cvt"
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
    if m: return float(m.group(1) or m.group(2))
    return np.nan

def build_features(brand, model_name, model_year, milage, fuel_type,
                   transmission, ext_col, accident, engine=""):
    car_age = CURRENT_YEAR - model_year
    fuel_remap = {"–": "unknown", "not supported": "unknown", "nan": "unknown", "": "unknown"}
    return pd.DataFrame([{
        "brand": brand, "model": model_name,
        "fuel_type": fuel_remap.get(fuel_type, fuel_type),
        "transmission_clean": simplify_transmission(transmission),
        "ext_col_clean": simplify_ext_col(ext_col),
        "model_year": model_year, "car_age": car_age, "milage": milage,
        "accident": int(accident),
        "engine_hp": extract_engine_hp(engine),
        "engine_cylinders": extract_cylinders(engine),
        "mileage_per_year": milage / (car_age + 1),
        "mileage_sq": milage ** 2,
        "age_sq": car_age ** 2,
        "age_x_mileage": car_age * milage,
    }])

def num_badge(n, filled=False):
    cls = "field-num-filled" if filled else "field-num"
    return f'<span class="{cls}">{n}</span>'


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
    data["milage"] = (data["milage"].astype(str)
        .str.replace(",", "", regex=False).str.extract(r"(\d+)"))
    data["milage"] = pd.to_numeric(data["milage"], errors="coerce")
    data["accident"] = data["accident"].fillna("none").astype(str).str.lower()
    data["accident"] = data["accident"].apply(
        lambda x: 0 if any(w in x for w in ["none", "no", "clean"]) else 1)
    return data

@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "rf_model.pkl"))

df = load_data()
ml_model = load_model()

# ----------------------------
# UI
# ----------------------------

brands = sorted(df["brand"].dropna().unique())

st.markdown("""
<div class="card">
  <div class="card-header">
    <div class="header-top">
      <div class="pill"><span class="pill-dot"></span>AI Powered</div>
      <div class="header-right">
        <span class="model-tag">XGBoost Model Accuracy</span>
        <div class="accuracy-badge"><span class="accuracy-num">88%</span></div>
      </div>
    </div>
    <div class="card-title">Used Car <em>Price</em><br/>Predictor</div>
    <p class="card-sub">Select a vehicle to get an instant market valuation</p>
    <div class="steps">
      <div class="step"></div>
      <div class="step"></div>
      <div class="step"></div>
    </div>
  </div>
  <div class="card-body">
""", unsafe_allow_html=True)

st.markdown(f'<div class="field"><div class="field-label">{num_badge(1)} Brand</div></div>', unsafe_allow_html=True)
brand = st.selectbox("brand_sel", [""] + brands,
    format_func=lambda x: "Select Brand" if x == "" else x.title(),
    label_visibility="collapsed")

models_list = sorted(df[df["brand"] == brand]["model"].unique()) if brand else []
st.markdown(f'<div class="field"><div class="field-label">{num_badge(2, bool(brand))} Model</div></div>', unsafe_allow_html=True)
model_name = st.selectbox("model_sel", [""] + models_list,
    format_func=lambda x: "Select Model" if x == "" else x.title(),
    disabled=not brand, label_visibility="collapsed")

years = sorted(df[(df["brand"] == brand) & (df["model"] == model_name)]["model_year"]
    .dropna().astype(int).unique()) if (brand and model_name) else []
st.markdown(f'<div class="field"><div class="field-label">{num_badge(3, bool(model_name))} Year</div></div>', unsafe_allow_html=True)
year = st.selectbox("year_sel", [""] + [str(y) for y in years],
    format_func=lambda x: "Select Year" if x == "" else x,
    disabled=not model_name, label_visibility="collapsed")

st.markdown('<hr class="divider-line">', unsafe_allow_html=True)

car_row = pd.DataFrame()
if brand and model_name and year:
    car_row = df[(df["brand"] == brand) & (df["model"] == model_name) & (df["model_year"] == int(year))]

if not car_row.empty:
    car          = car_row.iloc[0]
    milage       = float(car.get("milage", 0) or 0)
    fuel_type    = car.get("fuel_type", "gasoline")
    transmission = car.get("transmission", "automatic")
    ext_col      = car.get("ext_col", "other")
    accident     = int(car.get("accident", 0))
    engine       = car.get("engine", "")
    engine_hp    = extract_engine_hp(engine)
    engine_cyl   = extract_cylinders(engine)
    car_age      = CURRENT_YEAR - int(year)

    image_url = get_car_image(brand)
    st.markdown(f"""
    <div class="car-image-wrap">
      <img src="{image_url}" alt="{brand} car" />
      <div class="car-image-overlay"><span>{brand.title()}</span></div>
    </div>""", unsafe_allow_html=True)

    acc_class = "spec-val-red" if accident else "spec-val-green"
    acc_text  = "Reported" if accident else "None Reported"
    st.markdown(f"""
    <div class="specs-box">
      <div class="specs-title">Vehicle Specifications</div>
      <div class="spec-row"><span class="spec-key">⚙ Engine</span><span class="spec-val">{engine or "N/A"}</span></div>
      <div class="spec-row"><span class="spec-key">⚡ Transmission</span><span class="spec-val">{transmission.title()}</span></div>
      <div class="spec-row"><span class="spec-key">📍 Mileage</span><span class="spec-val">{int(milage):,} mi</span></div>
      <div class="spec-row"><span class="spec-key">📅 Vehicle Age</span><span class="spec-val">{car_age} {"year" if car_age == 1 else "years"}</span></div>
      <div class="spec-row"><span class="spec-key">⛽ Fuel Type</span><span class="spec-val">{fuel_type.title()}</span></div>
      <div class="spec-row"><span class="spec-key">🎨 Ext. Color</span><span class="spec-val">{ext_col.title()}</span></div>
      <div class="spec-row"><span class="spec-key">🛡 Accident</span><span class="{acc_class}">{acc_text}</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="refine-title">Refine Your Estimate</div>', unsafe_allow_html=True)

    fuel_options = ["gasoline", "hybrid", "plug-in hybrid", "diesel", "e85 flex fuel", "unknown"]
    fuel_default = fuel_type if fuel_type in fuel_options else "gasoline"
    st.markdown('<div class="field"><div class="field-label">⛽ Fuel Type</div></div>', unsafe_allow_html=True)
    fuel_sel = st.selectbox("fuel_sel", fuel_options,
        index=fuel_options.index(fuel_default), format_func=str.title, label_visibility="collapsed")

    trans_options = ["automatic", "manual", "cvt"]
    trans_default = simplify_transmission(transmission)
    st.markdown('<div class="field"><div class="field-label">⚡ Transmission</div></div>', unsafe_allow_html=True)
    trans_sel = st.selectbox("trans_sel", trans_options,
        index=trans_options.index(trans_default), format_func=str.title, label_visibility="collapsed")

    color_options = ["other", "black", "white", "gray", "silver", "blue", "red", "brown"]
    color_default = ext_col if ext_col in color_options else "other"
    st.markdown('<div class="field"><div class="field-label">🎨 Exterior Color</div></div>', unsafe_allow_html=True)
    color_sel = st.selectbox("color_sel", color_options,
        index=color_options.index(color_default), format_func=str.title, label_visibility="collapsed")

    st.markdown('<div class="field"><div class="field-label">🛡 Accident History</div></div>', unsafe_allow_html=True)
    acc_sel = st.selectbox("acc_sel", [0, 1], index=accident,
        format_func=lambda x: "No accidents reported" if x == 0 else "At least 1 accident reported",
        label_visibility="collapsed")

    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)

    if st.button("Get Price Estimate →"):
        X = build_features(
            brand=brand, model_name=model_name, model_year=int(year),
            milage=milage, fuel_type=fuel_sel, transmission=trans_sel,
            ext_col=color_sel, accident=acc_sel, engine=engine
        )
        price = np.expm1(ml_model.predict(X)[0])
        low, high = round(price * 0.92), round(price * 1.08)
        st.markdown(f"""
        <div class="result-panel">
          <div class="result-top">
            <div class="result-label">Estimated Market Value</div>
            <div class="result-price">${price:,.0f}</div>
            <div class="result-range">Market range: ${low:,} – ${high:,}</div>
          </div>
          <div class="result-bottom">
            <span class="result-dot"></span>
            <p class="result-note">Based on comparable listings. Actual price may vary by condition, location, and seller.</p>
          </div>
        </div>""", unsafe_allow_html=True)

elif brand and model_name and year:
    st.markdown('<p style="color:#f87171; font-size:13px; margin:0;">No data found for this selection.</p>', unsafe_allow_html=True)

st.markdown("""
  </div>
  <div class="card-footer">
    <span class="footer-dot"></span>
    <span class="footer-text">Powered by a trained XGBoost regression model &nbsp;·&nbsp; R² 0.8822</span>
  </div>
</div>
""", unsafe_allow_html=True)
