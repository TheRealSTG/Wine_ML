"""
app.py  —  WineScore Streamlit App
Run:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from functools import partial
import shap as _shap

st.set_page_config(page_title="WineScore", page_icon="🍷", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3,h4 { font-family: 'Cormorant Garamond', serif !important; letter-spacing: 0.02em; }
.stApp { background-color: #0c0808; color: #ede0d8; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#130a0a,#0f0606) !important;
    border-right: 1px solid rgba(180,60,60,0.2);
}
[data-testid="stSidebar"] .stMarkdown p { color: #c8b0a0; font-size:0.82rem; }
.stSlider > label { color:#d8c8b8 !important; font-size:0.79rem !important; }
div[data-testid="stSliderThumbValue"] { color:#f0c050 !important; font-size:0.75rem !important; }
[data-testid="stSidebar"] input[type="number"] {
    background:#1a0808 !important; border:1px solid #5a2020 !important;
    color:#f0e0d0 !important; border-radius:6px !important;
    font-size:0.78rem !important; padding:4px 6px !important; text-align:center;
}
[data-testid="stSidebar"] input[type="number"]:focus {
    border-color:#c05040 !important; box-shadow:0 0 0 2px rgba(192,80,64,0.2) !important;
}
[data-testid="stSidebar"] input[type="number"]::-webkit-inner-spin-button,
[data-testid="stSidebar"] input[type="number"]::-webkit-outer-spin-button { opacity:0.4; }
.stButton > button {
    background:transparent; border:1px solid #7a3030; color:#e0c8b8;
    border-radius:6px; font-family:'DM Sans',sans-serif; font-size:0.8rem; transition:all 0.2s;
}
.stButton > button:hover { background:#2a1010; border-color:#c05050; color:#f0e0d0; }
.stRadio > label { color:#c8b0a0 !important; font-size:0.78rem !important; }
hr { border-color:rgba(180,60,60,0.15) !important; margin:10px 0; }
.ws-card {
    background:linear-gradient(135deg,rgba(30,12,12,0.95),rgba(22,8,8,0.98));
    border:1px solid rgba(140,40,40,0.3); border-radius:14px; padding:28px 24px; margin:10px 0;
}
.section-label {
    font-size:0.65rem; font-weight:500; letter-spacing:0.15em; text-transform:uppercase;
    color:#c09878; margin-bottom:10px; padding-bottom:5px;
    border-bottom:1px solid rgba(180,60,60,0.22);
}
.badge-great { display:inline-block; padding:5px 18px; border-radius:30px;
  background:linear-gradient(135deg,#5a3a00,#3a2400); border:1px solid #d4a843;
  color:#d4a843; font-family:'Cormorant Garamond',serif; font-size:0.95rem; font-weight:600;
  letter-spacing:0.08em; text-transform:uppercase; }
.badge-avg { display:inline-block; padding:5px 18px; border-radius:30px;
  background:linear-gradient(135deg,#0e2010,#081408); border:1px solid #6a9a6a;
  color:#8aba8a; font-family:'Cormorant Garamond',serif; font-size:0.95rem; font-weight:600;
  letter-spacing:0.08em; text-transform:uppercase; }
.badge-poor { display:inline-block; padding:5px 18px; border-radius:30px;
  background:linear-gradient(135deg,#200808,#140404); border:1px solid #903030;
  color:#c06060; font-family:'Cormorant Garamond',serif; font-size:0.95rem; font-weight:600;
  letter-spacing:0.08em; text-transform:uppercase; }
[data-testid="metric-container"] {
    background:rgba(18,8,8,0.8) !important; border:1px solid rgba(100,30,30,0.3) !important;
    border-radius:10px !important; padding:12px 14px !important;
}
[data-testid="stMetricValue"] { color:#f0e0c8 !important; font-family:'Cormorant Garamond',serif !important; font-size:1.5rem !important; }
[data-testid="stMetricLabel"] { color:#c0a888 !important; font-size:0.68rem !important; text-transform:uppercase; letter-spacing:0.08em; }
.tip-card { background:rgba(18,8,8,0.7); border-left:3px solid #903030; border-radius:0 8px 8px 0; padding:11px 14px; margin:6px 0; }
.tip-card.tip-ok { border-left-color:#3a7a3a; }
.aroma-tag {
    display:inline-block; background:rgba(30,12,12,0.9); border:1px solid rgba(180,80,40,0.3);
    border-radius:20px; padding:4px 12px; margin:3px; font-size:0.76rem; color:#e0c8b0;
}
/* range warning pill */
.range-warn-edge  { display:inline-block; padding:1px 7px; border-radius:10px; font-size:0.6rem;
  background:rgba(180,60,20,0.2); border:1px solid rgba(180,60,20,0.5); color:#e08050;
  margin-left:6px; vertical-align:middle; }
.range-warn-near  { display:inline-block; padding:1px 7px; border-radius:10px; font-size:0.6rem;
  background:rgba(180,150,20,0.2); border:1px solid rgba(180,150,20,0.4); color:#d4b840;
  margin-left:6px; vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
QUALITY_COLORS = {0:"#c06060", 1:"#8aba8a", 2:"#d4a843"}
QUALITY_BADGES = {0:"badge-poor", 1:"badge-avg", 2:"badge-great"}
QUALITY_ICONS  = {0:"◈", 1:"◇", 2:"◆"}           # replaced bucket emoji
QUALITY_DESCS  = {
    0: "This wine's chemistry suggests notable flaws. See suggestions below.",
    1: "Solid, drinkable wine. A few tweaks could elevate it further.",
    2: "Excellent chemical profile — the hallmarks of a well-crafted wine.",
}

# (label, min, max, red_default, white_default, step, unit, desc, decimals)
FEATURE_META = {
    "fixed acidity":        ("Fixed Acidity",       4.0,   16.0,  7.4,   6.8,   0.1,   "g/L",  "Tartaric acid backbone",              1),
    "volatile acidity":     ("Volatile Acidity",    0.10,  1.60,  0.52,  0.28,  0.01,  "g/L",  "Acetic acid — high = vinegar taste",  2),
    "citric acid":          ("Citric Acid",          0.00,  1.00,  0.27,  0.33,  0.01,  "g/L",  "Freshness and citrus notes",          2),
    "residual sugar":       ("Residual Sugar",       1.0,   20.0,  2.5,   6.4,   0.1,   "g/L",  "Sugar remaining after fermentation",  1),
    "chlorides":            ("Chlorides",            0.010, 0.200, 0.080, 0.045, 0.005, "g/L",  "Salt content",                        3),
    "free sulfur dioxide":  ("Free SO₂",             1.0,   70.0,  14.0,  35.0,  1.0,   "mg/L", "Active preservative",                 0),
    "total sulfur dioxide": ("Total SO₂",            6.0,   300.0, 46.0,  138.0, 1.0,   "mg/L", "Free + bound SO₂",                    0),
    "density":              ("Density",              0.990, 1.004, 0.997, 0.994, 0.001, "g/cm³","Linked to sugar and alcohol",         3),
    "pH":                   ("pH",                   2.80,  4.00,  3.31,  3.19,  0.01,  "",     "Acidity scale",                       2),
    "sulphates":            ("Sulphates",            0.30,  2.00,  0.66,  0.49,  0.01,  "g/L",  "Antioxidant and preservative",        2),
    "alcohol":              ("Alcohol",              8.0,   15.0,  10.4,  10.5,  0.1,   "%",    "Alcohol by volume",                   1),
}

# Human-readable names for engineered features in SHAP chart
ENGINEERED_LABELS = {
    "so2_ratio":        "SO₂ Effectiveness",
    "alcohol_density":  "Fermentation Ratio",
    "acidity_ratio":    "Acid Balance",
    "total_acid":       "Total Acidity",
    "sugar_alcohol":    "Sweetness Ratio",
    "is_white":         "Wine Type",
}

def feat_display_name(f: str) -> str:
    if f in FEATURE_META:       return FEATURE_META[f][0]
    if f in ENGINEERED_LABELS:  return ENGINEERED_LABELS[f]
    return f.replace("_", " ").title()

# ── Range warning ─────────────────────────────────────────────────────────────
def range_status(val: float, mn: float, mx: float) -> str:
    """
    Returns 'edge' (within 5% of bounds), 'near' (5-12%), or 'ok'.
    Values outside bounds are allowed — model still runs but predictions
    may be less reliable.
    """
    span = mx - mn
    lo   = mn + span * 0.05
    hi   = mx - span * 0.05
    lo2  = mn + span * 0.12
    hi2  = mx - span * 0.12
    if val <= mn or val >= mx:   return "edge"
    if val < lo  or val > hi:    return "edge"
    if val < lo2 or val > hi2:   return "near"
    return "ok"

RANGE_PILL = {
    "edge": '<span class="range-warn-edge" title="Outside training range — predictions may be unreliable">⚠ out of range</span>',
    "near": '<span class="range-warn-near" title="Near the edge of the training data">~ boundary</span>',
    "ok":   "",
}

# ── Feature engineering (must exactly match train.py) ─────────────────────────
def engineer_features(vals: dict, wine_type_str: str) -> pd.DataFrame:
    df = pd.DataFrame([vals])
    df["so2_ratio"]       = (df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-6)).clip(0, 1)
    df["alcohol_density"] = df["alcohol"] / df["density"]
    df["acidity_ratio"]   = df["volatile acidity"] / (df["fixed acidity"] + 1e-6)
    df["total_acid"]      = df["fixed acidity"] + df["citric acid"]
    df["sugar_alcohol"]   = df["residual sugar"] / (df["alcohol"] + 1e-6)
    df["is_white"]        = int(wine_type_str == "white")
    return df

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model/pipeline.pkl")

@st.cache_resource
def get_explainer(_pipeline):
    clf  = _pipeline.named_steps["clf"]
    base = clf.calibrated_classifiers_[0].estimator if hasattr(clf, "calibrated_classifiers_") else clf
    return _shap.TreeExplainer(base, feature_perturbation="tree_path_dependent")

try:
    meta         = load_model()
    pipeline     = meta["pipeline"]
    FEATURE_COLS = meta["feature_cols"]
    LABEL_NAMES  = meta["label_names"]
    model_acc    = meta["accuracy"]
    model_fb     = meta["fbeta"]
    model_kappa  = meta.get("kappa", None)
    model_ver    = meta.get("model_version", "1.0.0")
    # Per-class decision thresholds (saved by train.py v2.1+)
    # Falls back to None for older models — argmax used in that case
    THRESHOLDS   = meta.get("thresholds", None)
except FileNotFoundError:
    st.error("⚠️  Model not found. Run `python train.py` first.")
    st.stop()

explainer = get_explainer(pipeline)

# ── Session state & reset ─────────────────────────────────────────────────────
if "wine_type" not in st.session_state:
    st.session_state.wine_type = "🍷 Red"

def get_default(feat: str, wt: str) -> float:
    _, _, _, rd, wd, *_ = FEATURE_META[feat]
    return wd if "White" in wt else rd

def reset_sliders():
    wt = st.session_state.get("wine_type_radio", "🍷 Red")
    st.session_state.wine_type = wt
    for feat in FEATURE_META:
        dec = FEATURE_META[feat][8]
        v   = get_default(feat, wt)
        v   = int(v) if dec == 0 else v
        st.session_state[f"sl_{feat}"] = v
        st.session_state[f"ni_{feat}"] = v

def slider_changed(feat: str):
    dec = FEATURE_META[feat][8]
    v   = st.session_state[f"sl_{feat}"]
    st.session_state[f"ni_{feat}"] = int(v) if dec == 0 else v

def input_changed(feat: str):
    _, mn, mx, _, _, _, _, _, dec = FEATURE_META[feat]
    raw = st.session_state[f"ni_{feat}"]
    # Allow slightly beyond bounds (for extrapolation exploration), but cap at ±20%
    span  = mx - mn
    hard_min = mn - span * 0.20
    hard_max = mx + span * 0.20
    v = float(max(hard_min, min(hard_max, raw)))
    st.session_state[f"sl_{feat}"] = max(mn, min(mx, v))  # slider stays within bounds
    # but store the actual typed value for prediction
    st.session_state[f"ni_{feat}"] = int(v) if dec == 0 else v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 14px 0;'>
      <div style='font-family:Cormorant Garamond,serif;font-size:1.55rem;color:#e8d0c0;font-weight:600;'>
        🍷 WineScore
      </div>
      <div style='font-size:0.76rem;color:#c8b0a0;margin-top:3px;line-height:1.5;'>
        Predict quality from chemical properties<br>UCI Vinho Verde dataset
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    wine_type = st.radio("Wine Type", ["🍷 Red", "🥂 White"],
                         horizontal=True, key="wine_type_radio")
    if wine_type != st.session_state.wine_type:
        st.session_state.wine_type = wine_type
        for feat in FEATURE_META:
            dec = FEATURE_META[feat][8]
            v   = get_default(feat, wine_type)
            v   = int(v) if dec == 0 else v
            st.session_state[f"sl_{feat}"] = v
            st.session_state[f"ni_{feat}"] = v

    st.divider()
    st.markdown('<div class="section-label">Chemical Properties</div>', unsafe_allow_html=True)

    input_vals: dict[str, float] = {}
    any_warnings = False

    for feat, (lbl, mn, mx, rd, wd, step, unit, desc, dec) in FEATURE_META.items():
        if f"sl_{feat}" not in st.session_state:
            v = get_default(feat, wine_type)
            st.session_state[f"sl_{feat}"] = int(v) if dec == 0 else v
        if f"ni_{feat}" not in st.session_state:
            st.session_state[f"ni_{feat}"] = st.session_state[f"sl_{feat}"]

        # Use the number input value (may be slightly outside slider bounds)
        current_val = float(st.session_state[f"ni_{feat}"])
        status      = range_status(current_val, mn, mx)
        if status != "ok":
            any_warnings = True

        # Label row with range indicator
        unit_str = f" ({unit})" if unit else ""
        pill     = RANGE_PILL[status]
        st.markdown(
            f'<div style="font-size:0.76rem;color:#d0c0b0;margin:8px 0 2px 0;line-height:1.4;">'
            f'{lbl}{unit_str}{pill}</div>',
            unsafe_allow_html=True,
        )

        col_sl, col_ni = st.columns([3, 1])
        if dec == 0:
            ni_min, ni_max, ni_step, fmt = int(mn), int(mx), int(step), "%d"
            if f"ni_{feat}" in st.session_state:
                st.session_state[f"ni_{feat}"] = int(st.session_state[f"ni_{feat}"])
        else:
            ni_min, ni_max, ni_step, fmt = float(mn), float(mx), float(step), f"%.{dec}f"

        with col_sl:
            st.slider("", mn, mx, step=float(step), key=f"sl_{feat}",
                      label_visibility="collapsed", help=desc,
                      on_change=partial(slider_changed, feat))
        with col_ni:
            st.number_input("", min_value=ni_min, max_value=ni_max,
                            step=ni_step, format=fmt, key=f"ni_{feat}",
                            label_visibility="collapsed",
                            on_change=partial(input_changed, feat))

        input_vals[feat] = float(st.session_state[f"ni_{feat}"])

    st.divider()

    # Global range warning banner
    if any_warnings:
        st.markdown("""
        <div style='padding:8px 10px;background:rgba(160,60,10,0.12);border:1px solid rgba(180,80,20,0.3);
                    border-radius:7px;font-size:0.72rem;color:#d09060;line-height:1.5;margin-bottom:8px;'>
          ⚠ One or more values are near or outside the training range.
          Predictions remain active but may be less reliable.
        </div>
        """, unsafe_allow_html=True)

    st.button("↺  Reset to defaults", on_click=reset_sliders, use_container_width=True)

    threshold_str = ""
    if THRESHOLDS:
        threshold_str = (f"<br>T(Poor) <span style='color:#c06060'>{THRESHOLDS[0]:.2f}</span> · "
                         f"T(Great) <span style='color:#d4a843'>{THRESHOLDS[2]:.2f}</span>")
    kappa_str = f"· κ <span style='color:#f0c050'>{model_kappa}</span>" if model_kappa else ""
    algo_str  = "LightGBM · Optuna · Calibrated" if model_ver >= "2.0.0" else "Random Forest · GridSearchCV"
    st.markdown(f"""
    <div style='margin-top:12px;padding:9px 11px;background:rgba(255,255,255,0.02);
         border-radius:8px;border:1px solid rgba(100,30,30,0.22);'>
      <div style='font-size:0.65rem;color:#c09070;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:4px;'>
        Model v{model_ver}
      </div>
      <div style='font-size:0.75rem;color:#c8b0a0;line-height:1.6;'>
        {algo_str}<br>
        Acc <span style='color:#f0c050'>{model_acc*100:.1f}%</span> ·
        F0.5 <span style='color:#f0c050'>{model_fb:.3f}</span>
        {kappa_str}
        {threshold_str}
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────
wt_str        = "white" if "White" in wine_type else "red"
engineered_df = engineer_features(input_vals, wt_str)
input_df      = engineered_df[FEATURE_COLS]

proba = pipeline.predict_proba(input_df)[0]

# Apply per-class thresholds if available (v2.1+ models), else fall back to argmax
if THRESHOLDS:
    # Priority: Poor > Great > Average
    # A wine that crosses the Poor threshold is called Poor regardless of Great probability
    if proba[0] >= THRESHOLDS[0]:
        pred_class = 0
    elif proba[2] >= THRESHOLDS[2]:
        pred_class = 2
    else:
        pred_class = 1
    using_thresholds = True
else:
    pred_class = int(np.argmax(proba))
    using_thresholds = False

confidence  = float(proba[pred_class])

scaler       = pipeline.named_steps["scaler"]
input_scaled = scaler.transform(input_df)

sv_raw = explainer.shap_values(input_scaled, check_additivity=False)
if isinstance(sv_raw, list):
    sv = np.array(sv_raw[pred_class]).flatten()
elif hasattr(sv_raw, "ndim") and sv_raw.ndim == 3:
    sv = sv_raw[0, :, pred_class]
else:
    sv = np.array(sv_raw).flatten()
n = len(FEATURE_COLS)
sv = sv[:n] if len(sv) > n else np.pad(sv, (0, n - len(sv)))

label      = LABEL_NAMES[pred_class]
badge_cls  = QUALITY_BADGES[pred_class]
main_color = QUALITY_COLORS[pred_class]

# Base-feature-only SHAP for Top Influences (engineered features not user-facing)
base_sv_df = pd.DataFrame({
    "feature": FEATURE_COLS,
    "shap":    sv,
    "is_base": [f in FEATURE_META for f in FEATURE_COLS],
}).query("is_base")

# ── Sommelier logic ───────────────────────────────────────────────────────────
def compute_taste_profile(v: dict, wt: str) -> dict:
    return dict(
        Sweetness  = min(10, max(0, (v["residual sugar"] - 1) / 19 * 10)),
        Acidity    = min(10, max(0, (4.0 - v["pH"]) / 1.2 * 5 + (v["fixed acidity"] - 4) / 12 * 5)),
        Body       = min(10, max(0, (v["alcohol"] - 8) / 7 * 10)),
        Fruitiness = min(10, max(0, v["citric acid"] * 5 + min(v["residual sugar"] / 6, 1) * 5)),
        Complexity = min(10, max(0, (v["sulphates"] - 0.3) / 1.7 * 7 + max(0, 1 - v["volatile acidity"] / 0.8) * 3)),
    )

def get_aroma_notes(v: dict, wt: str) -> list:
    is_white = "White" in wt
    notes, va, ca, rs, alc, sul, ph = [], v["volatile acidity"], v["citric acid"], \
        v["residual sugar"], v["alcohol"], v["sulphates"], v["pH"]
    if   va > 0.90: notes.append(("🫙", "Sharp / Pungent",  "Strong acetic, possibly over-fermented"))
    elif va > 0.60: notes.append(("🍂", "Earthy",           "Subtle barnyard and earthy undertones"))
    else:           notes.append(("✨", "Clean Nose",        "Fresh, no detectable off-aromas"))
    if   ca > 0.50: notes.append(("🍋", "Citrus",           "Bright lemon zest and lime"))
    elif ca > 0.25: notes.append(("🍊", "Fresh Fruit",      "Light citrus and crisp apple"))
    if   rs > 12:   notes.append(("🍯", "Honeyed",          "Rich honey, tropical and stone fruit"))
    elif rs > 5:    notes.append(("🍑", "Stone Fruit",       "Peach, apricot and ripe pear"))
    elif rs > 2:    notes.append(("🍎", "Apple / Pear",     "Crisp apple with light floral lift"))
    else:           notes.append(("🪨", "Mineral / Dry",    "Bone dry, flinty mineral character"))
    if sul > 0.90:  notes.append(("⛰️", "Mineral Depth",   "Smoky, savoury mineral complexity"))
    if alc > 13.5:  notes.append(("🔥", "Warming",          "Noticeable alcohol warmth, full presence"))
    if ph  < 3.10:  notes.append(("⚡", "Razor Acidity",    "Quite tart, mouth-watering freshness"))
    elif ph > 3.65: notes.append(("🧈", "Soft & Round",     "Low perceived acidity, buttery feel"))
    if not is_white and sul > 0.80 and va < 0.50:
        notes.append(("🍵", "Fine Tannins", "Structured tannins, good aging potential"))
    return notes

def get_verdict(v: dict, tp: dict, wt: str) -> str:
    body_d  = "full-bodied"   if tp["Body"]      > 6.5 else ("medium-bodied" if tp["Body"]      > 4 else "light-bodied")
    acid_d  = "vibrant"       if tp["Acidity"]   > 6.5 else ("moderate"      if tp["Acidity"]   > 4 else "soft")
    sweet_d = "off-dry"       if tp["Sweetness"] > 4.5 else ("semi-dry"      if tp["Sweetness"] > 2 else "dry")
    color   = "white" if "White" in wt else "red"
    out  = f"A {sweet_d}, {body_d} {color} wine with {acid_d} acidity. "
    out += ("The nose is clean and fruit-forward, showing good freshness. "
            if v["volatile acidity"] < 0.40 and v["citric acid"] > 0.30
            else ("The nose carries volatile character that may divide tasters. "
                  if v["volatile acidity"] > 0.70
                  else "The nose is approachable with balanced aromatic character. "))
    out += ("Expect generous fruit expression on the palate"
            if tp["Fruitiness"] > 6
            else ("Moderate fruit presence with structural backbone"
                  if tp["Fruitiness"] > 3
                  else "Restrained fruit with mineral and structural emphasis"))
    out += ", finishing with lasting complexity." if tp["Complexity"] > 6 else ", finishing clean and relatively short."
    return out

taste_profile = compute_taste_profile(input_vals, wine_type)
aroma_notes   = get_aroma_notes(input_vals, wine_type)
verdict       = get_verdict(input_vals, taste_profile, wine_type)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='padding:24px 8px 8px;border-bottom:1px solid rgba(140,40,40,0.15);margin-bottom:22px;'>
  <div style='display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;'>
    <span style='font-family:Cormorant Garamond,serif;font-size:1.9rem;color:#e8d0c0;font-weight:600;'>Wine Quality Analysis</span>
    <span style='font-size:0.7rem;color:#b08060;letter-spacing:0.14em;text-transform:uppercase;'>
      {"Red Wine" if "Red" in wine_type else "White Wine"} · Vinho Verde
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Row 1: Prediction + SHAP ──────────────────────────────────────────────────
col_l, col_r = st.columns([5, 7], gap="large")

with col_l:
    # Quality score card
    icon = "🍷" if pred_class == 2 else ("🫗" if pred_class == 1 else "🥀")
    st.markdown(f"""
    <div class="ws-card" style="text-align:center;padding:32px 20px 24px;">
      <div style="font-size:2.6rem;margin-bottom:10px;">{icon}</div>
      <div style="font-family:Cormorant Garamond,serif;font-size:2.3rem;font-weight:700;
                  color:{main_color};line-height:1.1;margin-bottom:10px;">{label}</div>
      <div class="{badge_cls}" style="margin-bottom:12px;">Quality</div>
      <div style="color:#b09080;font-size:0.8rem;margin-top:4px;">{confidence*100:.1f}% model confidence</div>
      <div style="margin-top:14px;padding-top:14px;border-top:1px solid rgba(140,40,40,0.15);
                  font-size:0.8rem;color:#c0a898;line-height:1.5;font-style:italic;">
        {QUALITY_DESCS[pred_class]}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bars
    st.markdown('<div class="section-label" style="margin-top:18px;">Probability Breakdown</div>', unsafe_allow_html=True)
    for i, cls_name in enumerate(["Poor", "Average", "Great"]):
        prob = proba[i]; c = QUALITY_COLORS[i]; w = max(int(prob * 100), 1)
        is_p = (i == pred_class)
        st.markdown(f"""
        <div style="margin:6px 0;padding:9px 13px;background:rgba(15,6,6,0.6);border-radius:8px;
                    border:1px solid {c + '33' if is_p else 'transparent'};">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
            <span style="font-size:0.78rem;color:{'#e8d0c0' if is_p else '#6a5040'};">
              {"▶ " if is_p else "   "}{cls_name}
            </span>
            <span style="font-size:0.9rem;color:{c};font-family:Cormorant Garamond,serif;">{prob*100:.1f}%</span>
          </div>
          <div style="background:rgba(40,15,15,0.8);border-radius:3px;height:4px;overflow:hidden;">
            <div style="width:{w}%;background:{c};height:4px;border-radius:3px;box-shadow:0 0 5px {c}55;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Input summary — show top 3 most impactful BASE features for this prediction
    st.markdown('<div class="section-label" style="margin-top:18px;">Most Impactful Inputs</div>', unsafe_allow_html=True)
    top3 = base_sv_df.reindex(base_sv_df["shap"].abs().sort_values(ascending=False).index).head(3)
    metric_cols = st.columns(3)
    for col, (_, row) in zip(metric_cols, top3.iterrows()):
        m    = FEATURE_META[row["feature"]]
        dec  = m[8]; unit = m[6]
        val  = input_vals[row["feature"]]
        vstr = f"{val:.{dec}f}" + (f" {unit}" if unit else "")
        direction = "↑" if row["shap"] > 0 else "↓"
        col.metric(m[0], vstr, direction, delta_color="normal" if row["shap"] > 0 else "inverse")

with col_r:
    # SHAP chart — use human-readable names, include engineered features
    st.markdown('<div class="section-label">Feature Impact  (SHAP)</div>', unsafe_allow_html=True)

    shap_df = pd.DataFrame({
        "feature": [feat_display_name(f) for f in FEATURE_COLS],
        "shap":    sv,
    }).sort_values("shap", key=abs, ascending=True).tail(10)

    bar_colors  = [QUALITY_COLORS[2] if v >= 0 else QUALITY_COLORS[0] for v in shap_df["shap"]]
    text_labels = [f" +{v:.3f}" if v >= 0 else f" {v:.3f}" for v in shap_df["shap"]]
    xpad        = max(abs(shap_df["shap"].max()), abs(shap_df["shap"].min())) * 0.40

    fig_shap = go.Figure(go.Bar(
        x=shap_df["shap"], y=shap_df["feature"], orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=text_labels, textposition="outside",
        textfont=dict(size=10, color="#c0a890"),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
    ))
    fig_shap.update_layout(
        xaxis=dict(range=[shap_df["shap"].min() - xpad, shap_df["shap"].max() + xpad],
                   gridcolor="rgba(60,20,20,0.4)", zerolinecolor="#5c2020", zerolinewidth=1.5,
                   tickfont=dict(size=9, color="#8a7060")),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10, color="#d0b8a8")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,5,5,0.6)",
        font=dict(color="#c8b0a0", family="DM Sans"),
        height=320, margin=dict(l=0, r=80, t=10, b=30), bargap=0.32,
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    radar_col, tips_col = st.columns([1, 1], gap="medium")

    with radar_col:
        st.markdown('<div class="section-label">Chemical Profile</div>', unsafe_allow_html=True)
        rf_feats = ["alcohol", "volatile acidity", "citric acid", "sulphates", "pH", "residual sugar"]
        norm_r   = [(input_vals[f] - FEATURE_META[f][1]) / (FEATURE_META[f][2] - FEATURE_META[f][1]) for f in rf_feats]
        rl = [FEATURE_META[f][0] for f in rf_feats] + [FEATURE_META[rf_feats[0]][0]]
        nr = norm_r + [norm_r[0]]
        rc = main_color
        fig_radar = go.Figure(go.Scatterpolar(
            r=nr, theta=rl, fill="toself",
            fillcolor=f"rgba({int(rc[1:3],16)},{int(rc[3:5],16)},{int(rc[5:7],16)},0.15)",
            line=dict(color=rc, width=2),
        ))
        fig_radar.update_layout(
            polar=dict(bgcolor="rgba(10,4,4,0.6)",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="rgba(80,20,20,0.3)",
                                tickfont=dict(size=7, color="#5a3828"), showticklabels=False),
                angularaxis=dict(gridcolor="rgba(80,20,20,0.3)", tickfont=dict(size=9, color="#d0b8a8"))),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans"),
            height=240, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tips_col:
        st.markdown('<div class="section-label">Top Influences</div>', unsafe_allow_html=True)
        # Only base features — engineered features aren't actionable by a user
        top_pos = base_sv_df.nlargest(2, "shap")
        top_neg = base_sv_df.nsmallest(2, "shap")
        for _, row in top_pos.iterrows():
            m = FEATURE_META[row["feature"]]
            val = input_vals[row["feature"]]
            st.markdown(f"""
            <div class="tip-card tip-ok">
              <div style="font-size:0.68rem;color:#6aba6a;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">↑ Boosting score</div>
              <div style="font-size:0.86rem;color:#d8f0d8;font-weight:500;">{m[0]}</div>
              <div style="font-size:0.72rem;color:#8aaa8a;margin-top:2px;">{val:.{m[8]}f} {m[6]} · SHAP +{row['shap']:.3f}</div>
            </div>""", unsafe_allow_html=True)
        for _, row in top_neg.iterrows():
            m = FEATURE_META[row["feature"]]
            val = input_vals[row["feature"]]
            direction = "lower" if val > m[3] else "higher"
            st.markdown(f"""
            <div class="tip-card">
              <div style="font-size:0.68rem;color:#d06060;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:2px;">↓ Dragging score</div>
              <div style="font-size:0.86rem;color:#f0d8d8;font-weight:500;">{m[0]}</div>
              <div style="font-size:0.72rem;color:#c09090;margin-top:2px;">{val:.{m[8]}f} {m[6]} · try {direction}</div>
            </div>""", unsafe_allow_html=True)

# ── Row 2: Sommelier ──────────────────────────────────────────────────────────
st.markdown('<div style="height:20px;"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-label">Sommelier\'s Observation</div>', unsafe_allow_html=True)

som_l, som_m, som_r = st.columns([4, 4, 5], gap="large")

with som_l:
    st.markdown('<div style="font-size:0.65rem;color:#c09878;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;">Taste Profile</div>', unsafe_allow_html=True)
    taste_dims = ["Sweetness", "Acidity", "Body", "Fruitiness", "Complexity"]
    taste_vals = [taste_profile[d] for d in taste_dims]
    fig_taste  = go.Figure(go.Scatterpolar(
        r=taste_vals + [taste_vals[0]], theta=taste_dims + [taste_dims[0]],
        fill="toself",
        fillcolor=f"rgba({int(main_color[1:3],16)},{int(main_color[3:5],16)},{int(main_color[5:7],16)},0.18)",
        line=dict(color=main_color, width=2.5),
        text=[f"{v:.1f}/10" for v in taste_vals + [taste_vals[0]]],
        hovertemplate="%{theta}: %{text}<extra></extra>",
    ))
    fig_taste.update_layout(
        polar=dict(bgcolor="rgba(10,4,4,0.6)",
            radialaxis=dict(visible=True, range=[0,10], gridcolor="rgba(80,20,20,0.3)",
                            tickfont=dict(size=8, color="#7a5848"), showticklabels=True, tickvals=[2,4,6,8,10]),
            angularaxis=dict(gridcolor="rgba(80,20,20,0.3)", tickfont=dict(size=10, color="#e0c8b0"))),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Cormorant Garamond"),
        height=260, margin=dict(l=20, r=20, t=16, b=16), showlegend=False,
    )
    st.plotly_chart(fig_taste, use_container_width=True)
    for dim, val in zip(taste_dims, taste_vals):
        w = int(val / 10 * 100)
        st.markdown(f"""
        <div style="margin:5px 0;">
          <div style="display:flex;justify-content:space-between;font-size:0.74rem;color:#c0a888;margin-bottom:2px;">
            <span>{dim}</span><span style="color:{main_color};font-variant-numeric:tabular-nums;">{val:.1f}</span>
          </div>
          <div style="background:rgba(40,15,15,0.8);border-radius:3px;height:3px;">
            <div style="width:{w}%;background:{main_color};height:3px;border-radius:3px;opacity:0.85;"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

with som_m:
    st.markdown('<div style="font-size:0.65rem;color:#c09878;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;">Aroma & Palate Notes</div>', unsafe_allow_html=True)
    for icon, name, desc in aroma_notes:
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:11px;padding:9px 0;border-bottom:1px solid rgba(100,30,30,0.12);">
          <div style="font-size:1.3rem;line-height:1;flex-shrink:0;margin-top:1px;">{icon}</div>
          <div>
            <div style="font-size:0.83rem;font-weight:500;color:#e8d8c8;margin-bottom:1px;">{name}</div>
            <div style="font-size:0.73rem;color:#a89880;line-height:1.4;">{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

with som_r:
    st.markdown('<div style="font-size:0.65rem;color:#c09878;text-transform:uppercase;letter-spacing:0.12em;margin-bottom:8px;">Tasting Verdict</div>', unsafe_allow_html=True)

    tags = []
    tp = taste_profile
    tags.append("Dry" if tp["Sweetness"] < 2 else ("Off-Dry" if tp["Sweetness"] < 5 else "Sweet"))
    tags.append("Full Body" if tp["Body"] > 6.5 else ("Medium Body" if tp["Body"] > 4 else "Light Body"))
    tags.append("High Acid" if tp["Acidity"] > 6.5 else ("Med Acid" if tp["Acidity"] > 4 else "Low Acid"))
    tags.append("Fruity" if tp["Fruitiness"] > 5 else "Mineral")
    tags.append("Complex" if tp["Complexity"] > 6 else "Simple")
    if input_vals["alcohol"] > 13:             tags.append("High ABV")
    if input_vals["volatile acidity"] < 0.35:  tags.append("Clean")

    tag_html = "".join([f'<span class="aroma-tag">{t}</span>' for t in tags])
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(28,12,8,0.9),rgba(18,6,6,0.95));
                border:1px solid rgba(160,80,40,0.22);border-radius:12px;padding:20px 18px;">
      <div style="font-family:Cormorant Garamond,serif;font-size:1.06rem;color:#e8d8c0;
                  line-height:1.8;font-style:italic;">"{verdict}"</div>
      <div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(140,60,20,0.18);">
        <div style="font-size:0.65rem;color:#c09060;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:7px;">Profile at a Glance</div>
        <div style="display:flex;flex-wrap:wrap;gap:5px;">{tag_html}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-size:0.65rem;color:#c09878;text-transform:uppercase;letter-spacing:0.12em;margin:16px 0 8px;">Food Pairing</div>', unsafe_allow_html=True)
    is_white = "White" in wine_type
    if is_white:
        pairings = (["🧀 Soft cheeses","🍰 Fruit desserts","🫚 Foie gras"] if tp["Sweetness"] > 5
                    else ["🐟 Seafood","🥗 Light salads","🍋 Ceviche"] if tp["Acidity"] > 6
                    else ["🍗 Roast chicken","🥘 Creamy pasta","🧅 Gruyère"])
    else:
        pairings = (["🥩 Red meat","🫙 Aged cheeses","🍖 Lamb"] if tp["Body"] > 6.5
                    else ["🍄 Mushroom dishes","🥩 Duck","🧄 Herb roasts"] if tp["Complexity"] > 6
                    else ["🍕 Pizza","🥩 Charcuterie","🍝 Tomato pasta"])
    for p in pairings:
        st.markdown(f'<div style="font-size:0.8rem;color:#c8b098;padding:5px 0;border-bottom:1px solid rgba(100,30,30,0.1);">{p}</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
algo_footer = "LightGBM · Optuna · CalibratedClassifierCV · SHAP" if model_ver >= "2.0.0" else "RandomForest · GridSearchCV · SHAP"
st.markdown(f"""
<div style='margin-top:36px;padding:14px 8px;border-top:1px solid rgba(140,40,40,0.12);
            display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;'>
  <span style='font-size:0.7rem;color:#7a5848;'>UCI ML Repository · Vinho Verde, Portugal · 6,497 wines</span>
  <span style='font-size:0.7rem;color:#7a5848;'>{algo_footer}</span>
</div>
""", unsafe_allow_html=True)
