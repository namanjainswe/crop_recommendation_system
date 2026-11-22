"""
AgriAI | Next-Gen Crop Recommendation Dashboard
========================================================
Ranked crop recommendation and real-time weather integration.
"""

import os
from typing import Optional, Dict, Tuple
import requests

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =============================================================================
# CONFIG & PAGE SETUP
# =============================================================================
st.set_page_config(
    page_title="AgriAI | Smart Crop Advisory System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
LABELS = {
    "N": "Nitrogen", "P": "Phosphorus", "K": "Potassium",
    "temperature": "Temperature", "humidity": "Humidity",
    "ph": "pH Level", "rainfall": "Rainfall",
}
UNITS = {
    "N": "mg/kg", "P": "mg/kg", "K": "mg/kg",
    "temperature": "°C", "humidity": "%", "ph": "", "rainfall": "mm",
}
TOP_N = 5

DATASET_PATHS = [
    "dataset/Crop_recommendation.csv",
    "Crop_recommendation.csv",
    "data/Crop_recommendation.csv",
    "../dataset/Crop_recommendation.csv",
]
MODEL_PATHS = {
    "Random Forest": ["models/random_forest.pkl", "random_forest.pkl", "../models/random_forest.pkl"],
    "Decision Tree": ["models/decision_tree.pkl", "decision_tree.pkl", "../models/decision_tree.pkl"],
}

# Design Tokens
BG = "#0B0F17"
SURFACE = "#131B26"
SURFACE_2 = "#1C2636"
BORDER = "#2A364B"
BORDER_LIGHT = "#3B4C66"
TEXT = "#F1F5F9"
MUTED = "#94A3B8"
GREEN = "#10B981"
GREEN_DIM = "#059669"
GREEN_DEEP = "#047857"
AMBER = "#F59E0B"
RED = "#EF4444"
BLUE = "#3B82F6"
PURPLE = "#8B5CF6"
RANK_COLORS = [GREEN, "#22C55E", "#10B981", "#059669", "#047857"]

# =============================================================================
# CUSTOM STYLING
# =============================================================================
st.markdown(f"""
<style>
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: {BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {MUTED}; }}

    .block-container {{ padding-top: 1.8rem; padding-bottom: 2rem; }}
    
    .agri-card {{
        background-color: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        margin-bottom: 16px;
    }}
    
    .stButton>button {{
        border-radius: 10px;
        transition: all 0.2s ease-in-out;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background-color: {SURFACE};
        padding: 6px;
        border-radius: 12px;
        border: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        color: {MUTED};
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {SURFACE_2} !important;
        color: {TEXT} !important;
    }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPER UI FUNCTIONS
# =============================================================================
def card(body: str, pad: str = "20px 24px", extra: str = "") -> str:
    return f'<div class="agri-card" style="padding:{pad};{extra}">{body}</div>'

def chip(text: str, color: str) -> str:
    return (f'<span style="display:inline-flex;align-items:center;background:{color}18;color:{color};'
            f'border:1px solid {color}40;border-radius:20px;padding:4px 12px;'
            f'font-size:12px;font-weight:600;margin:3px 6px 3px 0;white-space:nowrap;">'
            f'<span style="width:6px;height:6px;border-radius:50%;background:{color};margin-right:6px;"></span>{text}</span>')

def hero_card(row) -> str:
    conf = row["confidence"]
    conf_txt = "—" if pd.isna(conf) else f"{conf:.1f}%"
    fit = row["fit"]
    fit_txt = "—" if fit is None or pd.isna(fit) else f"{fit:.0f} / 100"
    pct = 0 if pd.isna(conf) else max(float(conf), 1)
    
    return card(f"""
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px;">
          <div style="min-width:220px;">
            <div style="color:{MUTED};font-size:11px;font-weight:700;letter-spacing:.08em;margin-bottom:6px;">PRIMARY RECOMMENDATION</div>
            <div style="font-size:38px;font-weight:800;color:{TEXT};letter-spacing:-0.02em;line-height:1.1;">{str(row['crop']).title()}</div>
          </div>
          <div style="display:flex;gap:36px;background:{SURFACE_2};padding:12px 24px;border-radius:12px;border:1px solid {BORDER_LIGHT};">
            <div>
              <div style="color:{MUTED};font-size:11px;font-weight:600;margin-bottom:2px;">MODEL CONFIDENCE</div>
              <div style="font-size:26px;font-weight:700;color:{GREEN};">{conf_txt}</div>
            </div>
            <div style="width:1px;background:{BORDER};margin:4px 0;"></div>
            <div>
              <div style="color:{MUTED};font-size:11px;font-weight:600;margin-bottom:2px;">SOIL FIT SCORE</div>
              <div style="font-size:26px;font-weight:700;color:{TEXT};">{fit_txt}</div>
            </div>
          </div>
        </div>
        <div style="margin-top:20px;">
          <div style="display:flex;justify-content:space-between;font-size:11px;color:{MUTED};margin-bottom:6px;">
             <span>Probability Match</span>
             <span>{conf_txt}</span>
          </div>
          <div style="background:{SURFACE_2};height:8px;border-radius:6px;overflow:hidden;border:1px solid {BORDER};">
            <div style="width:{pct:.1f}%;height:100%;background:linear-gradient(90deg,{GREEN_DEEP},{GREEN});border-radius:6px;"></div>
          </div>
        </div>
    """, pad="22px 26px")

def leaderboard(ranked: pd.DataFrame) -> str:
    rows = []
    for _, r in ranked.iloc[1:].iterrows():
        i = int(r["rank"])
        conf = r["confidence"]
        conf_txt = "—" if pd.isna(conf) else f"{conf:.1f}%"
        pct = 0 if pd.isna(conf) else max(float(conf), 1)
        color = RANK_COLORS[min(i - 1, len(RANK_COLORS) - 1)]
        fit = r["fit"]
        fit_txt = "" if fit is None or pd.isna(fit) else f"Soil Fit: <b>{fit:.0f}</b>"
        
        rows.append(f"""
        <div style="display:flex;align-items:center;gap:16px;padding:12px 0;border-top:1px solid {BORDER};">
          <div style="width:28px;height:28px;border-radius:8px;background:{SURFACE_2};display:flex;align-items:center;justify-content:center;color:{MUTED};font-size:13px;font-weight:700;">#{i}</div>
          <div style="flex:1;min-width:0;">
            <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">
              <span style="font-size:15px;font-weight:600;color:{TEXT};">{str(r['crop']).title()}</span>
              <span style="font-size:13px;font-weight:600;color:{TEXT};font-variant-numeric:tabular-nums;">{conf_txt}</span>
            </div>
            <div style="background:{SURFACE_2};height:6px;border-radius:4px;overflow:hidden;">
              <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:4px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;color:{MUTED};font-size:11px;margin-top:4px;">
               <span>{fit_txt}</span>
            </div>
          </div>
        </div>""")
        
    header = f'<div style="color:{MUTED};font-size:11px;font-weight:700;letter-spacing:.08em;padding-bottom:12px;">ALTERNATIVE CANDIDATES</div>'
    return card(header + "".join(rows), pad="18px 22px")

# =============================================================================
# DATA LOADERS & MODEL ENGINE
# =============================================================================
def first_existing(paths) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

@st.cache_resource(show_spinner=False)
def load_models():
    loaded, errors = {}, []
    for name, paths in MODEL_PATHS.items():
        path = first_existing(paths)
        if path is None:
            errors.append(f"{name}: not found ({paths[0]})")
            continue
        try:
            loaded[name] = joblib.load(path)
        except Exception as e:
            errors.append(f"{name}: {e}")
    return loaded, errors

@st.cache_data(show_spinner=False)
def load_dataset():
    path = first_existing(DATASET_PATHS)
    if path is None:
        return None, {}, {}, None, {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, {}, {}, None, {}

    label_col = "label" if "label" in df.columns else ("crop" if "crop" in df.columns else None)
    if label_col is None or [f for f in FEATURES if f not in df.columns]:
        return df, {}, {}, None, {}

    df["_key"] = df[label_col].astype(str).str.strip().str.lower()
    g = df.groupby("_key")[FEATURES]
    return (df, g.mean().to_dict("index"), g.std().fillna(0).to_dict("index"),
            label_col, {f: (float(df[f].min()), float(df[f].max())) for f in FEATURES})

models, model_errors = load_models()
dataset_df, BENCH, SPREAD, crop_col, BOUNDS = load_dataset()

if not BOUNDS:
    BOUNDS = {"N": (0, 140), "P": (0, 145), "K": (0, 205), "temperature": (8, 44),
              "humidity": (14, 100), "ph": (3.5, 9.9), "rainfall": (20, 300)}

def key_of(name) -> str:
    return str(name).strip().lower()

# =============================================================================
# WEATHER API INTEGRATION
# =============================================================================
def fetch_openweathermap_data(location_query: str, api_key: str) -> Tuple[Optional[Dict[str, float]], Optional[pd.DataFrame], Optional[str]]:
    """Fetch real-time weather and 7-day forecast from OpenWeatherMap."""
    if not api_key:
        demo_curr = {"temperature": 26.5, "humidity": 78.0, "rainfall": 125.0}
        dates = pd.date_range(end=pd.Timestamp.now(), periods=7).strftime("%b %d")
        demo_forecast = pd.DataFrame({
            "Date": dates,
            "Temp (°C)": [27.0, 29.5, 38.5, 32.0, 25.0, 24.5, 26.0],
            "Humidity (%)": [75, 70, 45, 60, 88, 92, 80],
            "Rainfall (mm)": [5.0, 0.0, 0.0, 12.0, 58.0, 22.0, 10.0]
        })
        return demo_curr, demo_forecast, "Using offline weather simulation (No API key provided)."

    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location_query}&limit=1&appid={api_key}"
        geo_res = requests.get(geo_url, timeout=5).json()
        
        if not geo_res or not isinstance(geo_res, list):
            return None, None, f"Location '{location_query}' not found."

        lat, lon = geo_res[0]["lat"], geo_res[0]["lon"]

        curr_url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={api_key}"
        curr_res = requests.get(curr_url, timeout=5).json()

        if str(curr_res.get("cod")) != "200":
            return None, None, f"Weather API Error: {curr_res.get('message', 'Failed to fetch weather data')}"

        curr_temp = float(curr_res["main"]["temp"])
        curr_humidity = float(curr_res["main"]["humidity"])
        curr_rain = float(curr_res.get("rain", {}).get("1h", 0.0)) * 24.0

        curr_data = {
            "temperature": round(curr_temp, 1),
            "humidity": round(curr_humidity, 1),
            "rainfall": round(max(curr_rain, 15.0), 1)
        }

        fore_url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={api_key}"
        fore_res = requests.get(fore_url, timeout=5).json()

        if str(fore_res.get("cod")) != "200":
            return curr_data, None, "Fetched current weather, but failed to load forecast."

        forecast_list = fore_res.get("list", [])
        daily_records = {}
        for entry in forecast_list:
            dt_txt = entry["dt_txt"].split(" ")[0]
            temp = entry["main"]["temp"]
            hum = entry["main"]["humidity"]
            rn = entry.get("rain", {}).get("3h", 0.0)

            if dt_txt not in daily_records:
                daily_records[dt_txt] = {"temps": [], "hums": [], "rains": []}
            daily_records[dt_txt]["temps"].append(temp)
            daily_records[dt_txt]["hums"].append(hum)
            daily_records[dt_txt]["rains"].append(rn)

        rows = []
        for d, vals in daily_records.items():
            rows.append({
                "Date": d,
                "Temp (°C)": round(float(np.max(vals["temps"])), 1),
                "Humidity (%)": round(float(np.mean(vals["hums"])), 1),
                "Rainfall (mm)": round(float(np.sum(vals["rains"])), 1)
            })

        return curr_data, pd.DataFrame(rows), None

    except Exception as ex:
        return None, None, f"Connection failed: {str(ex)}"

# =============================================================================
# AGRONOMIC SCORING LOGIC
# =============================================================================
def band_width(crop_key: str, f: str) -> float:
    sd = (SPREAD.get(crop_key) or {}).get(f, 0)
    if sd > 1e-9:
        return float(sd)
    lo, hi = BOUNDS[f]
    return max((hi - lo) * 0.05, 1e-6)

def agronomic_fit(reading: dict, crop_key: str) -> Optional[float]:
    bm = BENCH.get(crop_key)
    if not bm:
        return None
    z = [abs(reading[f] - bm[f]) / band_width(crop_key, f) for f in FEATURES]
    return float(np.clip(100 * np.exp(-0.45 * float(np.mean(z))), 0, 100))

def verdicts(reading: dict, crop_key: str) -> dict:
    bm = BENCH.get(crop_key)
    if not bm:
        return {}
    out = {}
    for f in FEATURES:
        z = (reading[f] - bm[f]) / band_width(crop_key, f)
        out[f] = {
            "z": z,
            "verdict": "fit" if abs(z) <= 1 else ("warn" if abs(z) <= 2 else "off"),
            "high": z > 0,
            "ideal": bm[f],
            "gap": reading[f] - bm[f],
        }
    return out

def rank_crops(model, reading: dict, top_n: int = TOP_N) -> pd.DataFrame:
    frame = pd.DataFrame([[reading[f] for f in FEATURES]], columns=FEATURES)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(frame)[0]
        order = np.argsort(probs)[::-1][:top_n]
        rows = [{"crop": str(model.classes_[i]), "confidence": float(probs[i]) * 100} for i in order]
    else:
        rows = [{"crop": str(model.predict(frame)[0]), "confidence": float("nan")}]
    for r in rows:
        r["fit"] = agronomic_fit(reading, key_of(r["crop"]))
    out = pd.DataFrame(rows)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out

# =============================================================================
# VISUALIZATIONS
# =============================================================================
def dark_layout(fig, height: int, **kw):
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="sans-serif", size=12),
        margin=dict(l=10, r=20, t=35, b=10),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER, font_color=TEXT),
        **kw,
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER, tickfont=dict(color=MUTED))
    return fig

def bullet_chart(reading: dict, crop_key: str) -> Optional[go.Figure]:
    bm = BENCH.get(crop_key)
    if not bm:
        return None

    v = verdicts(reading, crop_key)
    colors = {"fit": GREEN, "warn": AMBER, "off": RED}
    fig = make_subplots(rows=len(FEATURES), cols=1, vertical_spacing=0.08)

    for i, f in enumerate(FEATURES, start=1):
        lo, hi = BOUNDS[f]
        name = LABELS[f]
        w = band_width(crop_key, f)
        b_lo, b_hi = max(lo, bm[f] - w), min(hi, bm[f] + w)

        fig.add_trace(go.Bar(x=[hi - lo], y=[name], base=lo, orientation="h",
                             marker_color=SURFACE_2, width=0.5,
                             hoverinfo="skip", showlegend=False), row=i, col=1)
        fig.add_trace(go.Bar(x=[b_hi - b_lo], y=[name], base=b_lo, orientation="h",
                             marker_color=GREEN_DEEP, opacity=0.7, width=0.5,
                             name="Ideal Band (1 SD)", showlegend=(i == 1),
                             hovertemplate=f"{name} Ideal: {b_lo:.1f}–{b_hi:.1f} {UNITS[f]}<extra></extra>"),
                      row=i, col=1)
        fig.add_trace(go.Scatter(x=[reading[f]], y=[name], mode="markers",
                                 marker=dict(symbol="diamond", size=13,
                                             color=colors[v[f]["verdict"]],
                                             line=dict(width=2, color=BG)),
                                 name="Your Reading", showlegend=(i == 1),
                                 hovertemplate=f"{name}: %{{x:.1f}} {UNITS[f]}<extra></extra>"),
                      row=i, col=1)
        fig.update_xaxes(row=i, col=1, range=[lo, hi], showgrid=False, zeroline=False,
                         tickfont=dict(size=9, color=MUTED), linecolor=BORDER)
        fig.update_yaxes(row=i, col=1, showgrid=False, tickfont=dict(size=11, color=TEXT))

    dark_layout(fig, 48 * len(FEATURES) + 50, barmode="overlay",
                legend=dict(orientation="h", y=1.05, x=0, font=dict(size=11, color=MUTED)))
    return fig

# =============================================================================
# STATE MANAGEMENT
# =============================================================================
DEFAULTS = {
    "N": 90.0, "P": 42.0, "K": 43.0, "ph": 6.5,
    "temperature": 20.8, "humidity": 82.0, "rainfall": 202.9,
    "owm_key": "", "owm_city": "Delhi", "forecast_df": None,
    "ranked": None, "focus": None,
    "messages": [{"role": "assistant", "content":
                  "I'm your Agronomy Assistant. Ask me about fertilizer calculations, pH amendment strategies, or specific crop soil requirements."}],
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

def reading_now() -> dict:
    return {f: float(st.session_state[f]) for f in FEATURES}

# =============================================================================
# COPILOT ADVISORY ENGINE
# =============================================================================
def expert_reply(q_raw: str) -> str:
    q = q_raw.lower()
    r = reading_now()
    crop = st.session_state.focus
    label = str(crop).title() if crop else "your selected crop"
    k = key_of(crop) if crop else None
    bm = BENCH.get(k, {}) if k else {}

    if any(w in q for w in ["fertilizer", "urea", "dap", "mop", "deficien", "dose", "npk", "nutrient"]):
        if not bm:
            return "Run an analysis first and pick a crop candidate to calculate precise fertilizer dosage gaps."
        nd, pd_, kd = (max(0, bm["N"] - r["N"]), max(0, bm["P"] - r["P"]), max(0, bm["K"] - r["K"]))
        if nd == pd_ == kd == 0:
            return f"Your soil metrics already satisfy optimal N-P-K balances for **{label}**. No extra fertilizer required."
        return (f"### Fertilizer Recommendation for **{label}**\n\n"
                f"| Nutrient | Current Deficit | Recommended Fertilizer Amendment |\n|---|---|---|\n"
                f"| **Nitrogen (N)** | {nd:.1f} mg/kg | ~**{nd*2.17:.1f} kg/ha** Urea |\n"
                f"| **Phosphorus (P)** | {pd_:.1f} mg/kg | ~**{pd_*2.17:.1f} kg/ha** DAP |\n"
                f"| **Potassium (K)** | {kd:.1f} mg/kg | ~**{kd*1.67:.1f} kg/ha** MOP |\n\n"
                "**Application Note:** Split nitrogen applications across planting, tillering, and flowering stages. "
                "Always verify calculated rates against localized soil laboratory tests.")

    if any(w in q for w in ["ph", "acidic", "alkaline", "lime", "sulfur", "sulphur"]):
        ph = r["ph"]
        if ph < 5.5:
            return f"Your soil pH is **{ph} (Acidic)**. Apply Agricultural Lime (Calcium Carbonate) at 1–2 t/ha to bring it within 6.0–7.0. Reduce reliance on ammonium-based nitrogen sources."
        if ph > 7.8:
            return f"Your soil pH is **{ph} (Alkaline)**. Apply Elemental Sulfur or Gypsum to lower pH and increase micronutrient (Iron/Zinc) availability."
        return f"Your soil pH is **{ph}**, perfectly balanced within the optimal range (6.0–7.5) for standard crops."

    if any(w in q for w in ["water", "irrigat", "rainfall", "dry", "drought"]):
        rf = r["rainfall"]
        if rf < 60:
            return f"Rainfall is low (**{rf} mm**). Supplemental drip or sprinkler irrigation is recommended during initial establishment and flowering."
        if rf > 250:
            return f"Rainfall is high (**{rf} mm**). Ensure effective field drainage or employ raised beds to prevent root rot and waterlogging."
        return f"Rainfall (**{rf} mm**) is well-suited for regular development."

    return (f"Current soil readings: **N={r['N']}**, **P={r['P']}**, **K={r['K']}**, **pH={r['ph']}**, "
            f"**Temp={r['temperature']}°C**, **Humidity={r['humidity']}%**, **Rainfall={r['rainfall']}mm**.\n\n"
            f"Active Focus: **{label}**.")

# =============================================================================
# HEADER SECTION
# =============================================================================
ready = bool(models) and dataset_df is not None
dot = GREEN if ready else RED
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;
            gap:16px;padding:8px 0 20px 0;border-bottom:1px solid {BORDER};margin-bottom:24px;">
  <div style="display:flex;align-items:center;gap:14px;">
    <div style="width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,{GREEN_DEEP},{GREEN});
                display:flex;align-items:center;justify-content:center;font-size:22px;box-shadow:0 4px 12px rgba(16,185,129,0.25);">🌾</div>
    <div>
      <div style="font-size:22px;font-weight:800;color:{TEXT};letter-spacing:-0.02em;">AgriAI Advisory Platform</div>
      <div style="font-size:12px;color:{MUTED};">Precision crop recommendation engine powered by ML &amp; soil chemistry</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:8px;background:{SURFACE};padding:6px 14px;border-radius:20px;border:1px solid {BORDER};font-size:12px;color:{TEXT}; font-weight: 500;">
    <span style="width:8px;height:8px;border-radius:50%;background:{dot};display:inline-block;box-shadow: 0 0 8px {dot};"></span>
    {'System Operational' if ready else 'Setup Required'}
  </div>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown(f'<div style="font-size:12px;font-weight:700;color:{MUTED};letter-spacing:0.05em;margin-bottom:10px;">MODEL SELECTION</div>', unsafe_allow_html=True)
    if models:
        model_name = st.selectbox("Active Classifier", list(models.keys()), label_visibility="collapsed")
        active = models[model_name]
    else:
        model_name, active = None, None
        st.error("No model found. Place trained `.pkl` models in `models/` directory.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;font-weight:700;color:{MUTED};letter-spacing:0.05em;margin-bottom:10px;">CLIMATE PRESETS</div>', unsafe_allow_html=True)
    
    presets = {
        "🌴 Tropical": {"N": 90.0, "P": 45.0, "K": 40.0, "ph": 5.8, "temperature": 27.0, "humidity": 82.0, "rainfall": 220.0},
        "🌵 Arid": {"N": 20.0, "P": 25.0, "K": 30.0, "ph": 7.8, "temperature": 35.0, "humidity": 25.0, "rainfall": 30.0},
        "🏔️ Alpine": {"N": 20.0, "P": 130.0, "K": 200.0, "ph": 6.4, "temperature": 21.0, "humidity": 90.0, "rainfall": 110.0},
        "🌾 Savanna": {"N": 50.0, "P": 40.0, "K": 20.0, "ph": 6.2, "temperature": 29.0, "humidity": 50.0, "rainfall": 75.0},
    }
    ca, cb = st.columns(2)
    for i, (nm, vals) in enumerate(presets.items()):
        with (ca if i % 2 == 0 else cb):
            if st.button(nm, use_container_width=True):
                st.session_state.update(vals)
                st.session_state.ranked = None
                st.rerun()

    if st.button("🔄 Reset Parameters", use_container_width=True):
        for f in FEATURES:
            st.session_state[f] = DEFAULTS[f]
        st.session_state.ranked = None
        st.session_state.focus = None
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div style="font-size:12px;font-weight:700;color:{MUTED};letter-spacing:0.05em;margin-bottom:10px;">DATASET METRICS</div>', unsafe_allow_html=True)
    
    rows = [
        ("Models Loaded", str(len(models))),
        ("Active Model", model_name or "None"),
        ("Dataset Records", f"{len(dataset_df):,}" if dataset_df is not None else "Missing"),
        ("Crop Classes", str(dataset_df["_key"].nunique()) if (dataset_df is not None and crop_col) else "—"),
    ]
    st.markdown("".join(
        f'<div style="display:flex;justify-content:space-between;padding:8px 0;'
        f'border-bottom:1px solid {BORDER};font-size:12px;">'
        f'<span style="color:{MUTED};">{a}</span><span style="color:{TEXT};font-weight:600;">{b}</span></div>'
        for a, b in rows), unsafe_allow_html=True)

    for e in model_errors:
        st.caption(f"⚠️ {e}")

# =============================================================================
# TABS & NAVIGATION
# =============================================================================
t_rec, t_batch, t_crops, t_model, t_chat = st.tabs([
    "🌱 Recommendation",
    "📁 Batch Analysis",
    "📊 Crop Profiles",
    "🤖 Model Evaluation",
    "💬 Agronomy Copilot"
])

# ------------------------------ RECOMMEND TAB --------------------------------
with t_rec:
    with st.expander("🌤️ Real-Time Weather Auto-Fetch (OpenWeatherMap Integration)", expanded=True):
        wc1, wc2, wc3 = st.columns([2, 2, 1])
        with wc1:
            st.session_state.owm_city = st.text_input("City / Coordinates", value=st.session_state.owm_city)
        with wc2:
            st.session_state.owm_key = st.text_input("OpenWeather API Key (Optional)", value=st.session_state.owm_key, type="password")
        with wc3:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("🌐 Fetch Weather", use_container_width=True):
                with st.spinner("Fetching weather parameters..."):
                    w_curr, w_fore, w_err = fetch_openweathermap_data(st.session_state.owm_city, st.session_state.owm_key)
                    if w_curr:
                        st.session_state.temperature = w_curr["temperature"]
                        st.session_state.humidity = w_curr["humidity"]
                        st.session_state.rainfall = w_curr["rainfall"]
                        st.session_state.forecast_df = w_fore
                        if w_err:
                            st.info(w_err)
                        else:
                            st.success(f"Updated parameters for {st.session_state.owm_city}!")
                    else:
                        st.error(w_err)

    if st.session_state.forecast_df is not None:
        f_df = st.session_state.forecast_df
        max_t = f_df["Temp (°C)"].max()
        max_r = f_df["Rainfall (mm)"].max()
        
        alerts = []
        if max_t >= 38.0:
            alerts.append(chip(f"Heatwave Alert: Max Temp {max_t}°C Expected", RED))
        if max_r >= 50.0:
            alerts.append(chip(f"Heavy Rainfall Warning: Max Rain {max_r}mm Expected", AMBER))
            
        if alerts:
            st.markdown("".join(alerts), unsafe_allow_html=True)

        with st.expander("📅 View 7-Day Sowing Window Weather Outlook"):
            fig_fore = px.line(f_df, x="Date", y=["Temp (°C)", "Rainfall (mm)"],
                               markers=True, color_discrete_sequence=[AMBER, BLUE])
            dark_layout(fig_fore, 250)
            st.plotly_chart(fig_fore, use_container_width=True, config={"displayModeBar": False})

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f'<div style="color:{MUTED};font-size:11px;font-weight:700;letter-spacing:.08em;margin-bottom:8px;">SOIL CHEMISTRY INPUTS</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.number_input("Nitrogen (N) — mg/kg", 0.0, 200.0, key="N", step=1.0)
            st.number_input("Phosphorus (P) — mg/kg", 0.0, 200.0, key="P", step=1.0)
            st.number_input("Potassium (K) — mg/kg", 0.0, 300.0, key="K", step=1.0)
            st.number_input("pH Level", 0.0, 14.0, key="ph", step=0.1)
    with c2:
        st.markdown(f'<div style="color:{MUTED};font-size:11px;font-weight:700;letter-spacing:.08em;margin-bottom:8px;">CLIMATE INPUTS</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.number_input("Temperature (°C)", -10.0, 60.0, key="temperature", step=0.5)
            st.number_input("Humidity (%)", 0.0, 100.0, key="humidity", step=1.0)
            st.number_input("Rainfall (mm)", 0.0, 5000.0, key="rainfall", step=5.0)

    st.write("")
    if st.button("🚀 Analyze Soil & Rank Recommendation", type="primary",
                 disabled=active is None, use_container_width=True):
        try:
            st.session_state.ranked = rank_crops(active, reading_now(), TOP_N)
            st.session_state.focus = st.session_state.ranked.iloc[0]["crop"]
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.session_state.ranked = None

    ranked = st.session_state.ranked
    if ranked is not None and len(ranked):
        st.write("")
        st.markdown(hero_card(ranked.iloc[0]), unsafe_allow_html=True)

        if len(ranked) > 1:
            st.write("")
            st.markdown(leaderboard(ranked), unsafe_allow_html=True)

        st.write("")
        st.markdown(f'<div style="color:{MUTED};font-size:11px;font-weight:700;letter-spacing:.08em;margin-bottom:8px;">CANDIDATE DIAGNOSTICS & COMPARISON</div>', unsafe_allow_html=True)
        choice = st.radio("Candidate Selection", [str(c).title() for c in ranked["crop"]],
                          horizontal=True, label_visibility="collapsed")
        st.session_state.focus = choice
        k = key_of(choice)
        row = ranked[ranked["crop"].astype(str).str.lower() == k]

        if len(row):
            a, b, c = st.columns(3)
            a.metric("Rank Position", f"#{int(row.iloc[0]['rank'])}")
            cf = row.iloc[0]["confidence"]
            b.metric("Confidence", "—" if pd.isna(cf) else f"{cf:.1f}%")
            ft = row.iloc[0]["fit"]
            c.metric("Soil Fit Score", "—" if ft is None or pd.isna(ft) else f"{ft:.0f} / 100")

        r = reading_now()
        fig = bullet_chart(r, k)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            v = verdicts(r, k)
            flagged = [f for f in FEATURES if v[f]["verdict"] in ("off", "warn")]
            if flagged:
                chips = "".join(
                    chip(f"{LABELS[f]} {'High' if v[f]['high'] else 'Low'} by "
                         f"{abs(v[f]['gap']):.1f} {UNITS[f]}",
                         RED if v[f]["verdict"] == "off" else AMBER)
                    for f in flagged)
                st.markdown(f'<div style="color:{MUTED};font-size:11px;font-weight:600;margin-bottom:6px;">'
                            f'PARAMETERS OUTSIDE OPTIMAL BAND</div>{chips}', unsafe_allow_html=True)
            else:
                st.markdown(chip(f"All 7 parameters within optimal band for {choice}", GREEN),
                            unsafe_allow_html=True)

        st.write("")
        exp = ranked.copy()
        exp.columns = ["Rank", "Crop", "Confidence_pct", "Soil_Fit"]
        for f in FEATURES:
            exp[f] = st.session_state[f]
        st.download_button("📥 Export Recommendation Report (CSV)", exp.to_csv(index=False).encode("utf-8"),
                           "crop_ranking.csv", "text/csv")

# ------------------------------ BATCH TAB ------------------------------------
with t_batch:
    st.markdown("Upload a CSV dataset containing parameters: " + ", ".join(f"`{f}`" for f in FEATURES))
    up = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if up:
        try:
            bdf = pd.read_csv(up)
            st.dataframe(bdf.head(5), use_container_width=True)
            missing = [c for c in FEATURES if c not in bdf.columns]
            if missing:
                st.error(f"Missing required columns: {', '.join(missing)}")
            elif active is not None:
                n_alt = st.slider("Ranked choices per row", 1, TOP_N, 3)
                if st.button("Run Batch Inference", type="primary"):
                    clean = bdf.copy()
                    clean[FEATURES] = clean[FEATURES].apply(pd.to_numeric, errors="coerce")
                    clean = clean[clean[FEATURES].notna().all(axis=1)].reset_index(drop=True)

                    with st.spinner(f"Processing {len(clean)} records..."):
                        X = clean[FEATURES]
                        if hasattr(active, "predict_proba"):
                            P = active.predict_proba(X)
                            order = np.argsort(P, axis=1)[:, ::-1][:, :n_alt]
                            for j in range(n_alt):
                                idx = order[:, j]
                                clean[f"Choice_{j+1}"] = [str(active.classes_[i]).title() for i in idx]
                                clean[f"Choice_{j+1}_%"] = [round(P[rr, i] * 100, 1) for rr, i in enumerate(idx)]
                        else:
                            clean["Choice_1"] = active.predict(X)

                    st.success(f"Batch scoring completed for {len(clean)} rows.")
                    g1, g2 = st.columns([2, 1])
                    with g1:
                        st.dataframe(clean, use_container_width=True)
                        st.download_button("📥 Download Results (CSV)",
                                           clean.to_csv(index=False).encode("utf-8"),
                                           "batch_predictions.csv", "text/csv")
                    with g2:
                        dist = clean["Choice_1"].value_counts()
                        fig = px.pie(values=dist.values, names=dist.index, hole=0.55,
                                     color_discrete_sequence=[GREEN, BLUE, AMBER, RED, PURPLE])
                        dark_layout(fig, 300, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.error(f"Could not process file: {e}")

# ------------------------------ CROP PROFILES TAB ----------------------------
with t_crops:
    if BENCH:
        names = sorted(BENCH.keys())
        sel = st.multiselect("Select Crops to Compare", [n.title() for n in names],
                             default=[n.title() for n in names[:4]], max_selections=6)
        if sel:
            keys = [key_of(s) for s in sel]
            prof = pd.DataFrame({s: BENCH[k] for s, k in zip(sel, keys)}).T.round(1)
            prof.columns = [f"{LABELS[c]} ({UNITS[c]})".strip() for c in prof.columns]
            st.dataframe(prof, use_container_width=True)

            st.write("")
            f = st.selectbox("Parameter Comparison", FEATURES, format_func=lambda x: LABELS[x])
            r = reading_now()
            fig = go.Figure(go.Bar(
                x=sel, y=[BENCH[k][f] for k in keys], marker_color=GREEN,
                error_y=dict(type="data", array=[SPREAD[k][f] for k in keys],
                             color=MUTED, thickness=1.5, width=6)))
            fig.add_hline(y=r[f], line_dash="dash", line_color=AMBER,
                          annotation_text=f"Your Reading: {r[f]:.1f} {UNITS[f]}",
                          annotation_position="top left")
            dark_layout(fig, 360, yaxis_title=f"{LABELS[f]} ({UNITS[f]})".strip())
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ------------------------------ MODEL EVALUATION TAB ------------------------
with t_model:
    if dataset_df is not None and active is not None and crop_col:
        X, y = dataset_df[FEATURES], dataset_df[crop_col]
        try:
            preds = active.predict(X)
            a, b, c, d = st.columns(4)
            a.metric("Model Accuracy", f"{accuracy_score(y, preds)*100:.2f}%")
            b.metric("Total Records", f"{len(X):,}")
            c.metric("Classes", f"{y.nunique()}")
            d.metric("Active Model", model_name)

            with st.expander("Detailed Classification Metrics"):
                rep = pd.DataFrame(classification_report(y, preds, output_dict=True, zero_division=0)).T.round(3)
                st.dataframe(rep[~rep.index.isin(["accuracy"])], use_container_width=True)

            st.write("")
            norm = st.toggle("Normalize Confusion Matrix (%)")
            cm = confusion_matrix(y, preds, labels=active.classes_)
            cm_show = (cm / cm.sum(axis=1, keepdims=True) * 100) if norm else cm
            fig = px.imshow(cm_show, x=active.classes_, y=active.classes_,
                            color_continuous_scale=[[0, SURFACE], [1, GREEN]])
            dark_layout(fig, 520, xaxis_title="Predicted Class", yaxis_title="Actual Class")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.error(f"Evaluation error: {e}")

# ------------------------------ COPILOT CHAT TAB -----------------------------
with t_chat:
    st.markdown(f'<div style="color:{MUTED};font-size:12px;margin-bottom:12px;">SUGGESTED PROMPTS</div>', unsafe_allow_html=True)
    q1, q2, q3 = st.columns(3)
    if q1.button("💡 Calculate Fertilizer Plan", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "How much fertilizer should I apply?"})
        st.rerun()
    if q2.button("🧪 Soil pH Correction Guidance", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "How do I fix my soil pH?"})
        st.rerun()
    if q3.button("⚖️ Compare Reading to Ideal Profile", use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": "Compare my soil to this crop"})
        st.rerun()

    st.write("")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if prompt := st.chat_input("Ask Agronomy Assistant..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        reply = expert_reply(prompt)
        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

    if len(st.session_state.messages) > 1:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = DEFAULTS["messages"]
            st.rerun()
