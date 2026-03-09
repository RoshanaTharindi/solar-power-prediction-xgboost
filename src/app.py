# src/app.py
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

# =========================
# Page config
# =========================
st.set_page_config(page_title="SolarCast Sri Lanka", page_icon="☀️", layout="wide")

# =========================
# Background (local image) + Animated Sun Glow + Commercial UI CSS
# =========================
def set_background_image(img_path: str):
    """
    Uses a local background image (recommended: assets/solar_bg.jpg).
    If missing, falls back to a dark gradient.
    """
    p = Path(img_path)
    if p.exists():
        b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
        bg_css = f"""
        <style>
        .stApp {{
          background-image:
            linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
            url("data:image/jpg;base64,{b64}");
          background-size: cover;
          background-position: center;
          background-attachment: fixed;
          background-repeat: no-repeat;
        }}
        </style>
        """
    else:
        bg_css = """
        <style>
        .stApp {
          background: radial-gradient(circle at 20% 10%, rgba(255,184,0,0.20), transparent 45%),
                      radial-gradient(circle at 70% 25%, rgba(0,188,212,0.14), transparent 45%),
                      linear-gradient(180deg, rgba(10,12,16,1), rgba(14,16,20,1));
        }
        </style>
        """
    st.markdown(bg_css, unsafe_allow_html=True)

set_background_image("assets/solar_bg.jpg")

st.markdown(
    """
<style>
/* Layout polish */
.block-container { padding-top: 1.1rem; padding-bottom: 1.5rem; max-width: 1200px; }

/* Hero */
.hero {
  padding: 18px 18px 14px 18px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(255, 184, 0, 0.16), rgba(0, 188, 212, 0.10));
  border: 1px solid rgba(255,255,255,0.08);
  position: relative;
  overflow: hidden;
}
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.08);
}

/* Cards */
.card {
  padding: 16px;
  border-radius: 16px;
  background: rgba(20,20,20,0.85);
  border: 1px solid rgba(255,255,255,0.07);
  backdrop-filter: blur(6px);
}
.small { opacity: 0.85; font-size: 0.92rem; }
.tiny  { opacity: 0.72; font-size: 0.85rem; }

/* Animated Sun Glow */
.sun-glow {
  position: fixed;
  top: -120px;
  left: -120px;
  width: 420px;
  height: 420px;
  border-radius: 999px;
  background: radial-gradient(circle,
    rgba(255, 214, 102, 0.95) 0%,
    rgba(255, 153, 0, 0.35) 35%,
    rgba(255, 153, 0, 0.10) 60%,
    rgba(255, 153, 0, 0.00) 75%);
  filter: blur(2px);
  z-index: 0;
  pointer-events: none;
  animation: pulseGlow 4.5s ease-in-out infinite;
}
@keyframes pulseGlow {
  0%   { transform: scale(0.95); opacity: 0.55; }
  50%  { transform: scale(1.05); opacity: 0.80; }
  100% { transform: scale(0.95); opacity: 0.55; }
}

/* Make content above the glow */
.stApp > header, .stApp > div { position: relative; z-index: 1; }

/* Gauge (PV meter with needle animation) */
.gauge-wrap { display: flex; gap: 18px; align-items: center; }
.gauge {
  width: 260px;
  height: 150px;
  position: relative;
}
.gauge .arc {
  width: 260px;
  height: 130px;
  border-top-left-radius: 260px;
  border-top-right-radius: 260px;
  background: conic-gradient(
    #4CAF50 0deg,
    #FFC107 120deg,
    #F44336 240deg,
    #F44336 360deg
  );
  /* show only top half */
  clip-path: inset(0 0 50% 0);
  opacity: 0.95;
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow: 0 10px 40px rgba(0,0,0,0.35);
}
.gauge .mask {
  width: 220px;
  height: 110px;
  position: absolute;
  left: 20px;
  top: 20px;
  background: rgba(20,20,20,0.92);
  border-top-left-radius: 220px;
  border-top-right-radius: 220px;
  clip-path: inset(0 0 50% 0);
  border: 1px solid rgba(255,255,255,0.08);
}
.gauge .center {
  width: 16px;
  height: 16px;
  background: rgba(255,255,255,0.92);
  border-radius: 999px;
  position: absolute;
  left: calc(50% - 8px);
  top: 82px;
  box-shadow: 0 0 0 6px rgba(255,184,0,0.15);
}
.gauge .needle {
  width: 3px;
  height: 78px;
  background: rgba(255,255,255,0.95);
  position: absolute;
  left: calc(50% - 1.5px);
  top: 20px;
  transform-origin: bottom center;
  border-radius: 8px;
  box-shadow: 0 6px 16px rgba(0,0,0,0.35);
  transition: transform 900ms cubic-bezier(.22,.61,.36,1);
}
.gauge .label {
  position: absolute;
  width: 100%;
  text-align: center;
  top: 110px;
  font-weight: 600;
}
.gauge .sub {
  position: absolute;
  width: 100%;
  text-align: center;
  top: 132px;
  font-size: 0.85rem;
  opacity: 0.75;
}

/* Buttons */
div.stButton > button {
  border-radius: 12px;
  padding: 0.70rem 1rem;
}

/* Remove default Streamlit chrome spacing */
hr { border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 12px 0; }
</style>

<div class="sun-glow"></div>
""",
    unsafe_allow_html=True,
)

# =========================
# Defaults (no sidebar)
# =========================
DATA_PATH = "data/processed.csv"
MODEL_PATH = "models/xgb_model.json"

# =========================
# Caching
# =========================
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["timestamp"])

@st.cache_resource(show_spinner=False)
def load_model(path: str) -> XGBRegressor:
    m = XGBRegressor()
    m.load_model(path)
    return m

# =========================
# Helpers
# =========================
def add_time_features_to_row(row: dict, ts: pd.Timestamp):
    row["year"] = ts.year
    row["month"] = ts.month
    row["day"] = ts.day
    row["hour"] = ts.hour
    row["dayofyear"] = ts.dayofyear
    row["weekday"] = ts.weekday()

    row["hour_sin"] = float(np.sin(2 * np.pi * row["hour"] / 24.0))
    row["hour_cos"] = float(np.cos(2 * np.pi * row["hour"] / 24.0))
    row["doy_sin"] = float(np.sin(2 * np.pi * row["dayofyear"] / 365.25))
    row["doy_cos"] = float(np.cos(2 * np.pi * row["dayofyear"] / 365.25))
    return row

def build_features_row(df: pd.DataFrame, feature_cols, city: str, ts: pd.Timestamp,
                       T2M: float, RH2M: float, WS2M: float):
    row = {c: 0 for c in feature_cols}

    # Weather drivers
    for k, v in {"T2M": T2M, "RH2M": RH2M, "WS2M": WS2M}.items():
        if k in row:
            row[k] = float(v)

    # City one-hot
    city_col = f"city_{city}"
    if city_col in row:
        row[city_col] = 1.0

    # Lat/Lon from dataset median for that city (if present)
    if city_col in df.columns:
        sub = df[df[city_col] == 1]
        if len(sub):
            if "lat" in row and "lat" in sub.columns:
                row["lat"] = float(sub["lat"].median())
            if "lon" in row and "lon" in sub.columns:
                row["lon"] = float(sub["lon"].median())

    # Time features
    row = add_time_features_to_row(row, ts)
    return row

def label_intensity(irr: float, max_irr: float = 1000.0):
    pct = float(np.clip(irr / max_irr, 0.0, 1.0))
    if pct < 0.30:
        return "Low"
    if pct < 0.60:
        return "Moderate"
    return "High"

def render_gauge(title: str, value: float, unit: str, vmin: float, vmax: float, subtitle: str = ""):
    """
    Semi-circle gauge with a needle that animates to the value.
    Needle rotation range: -90deg (min) to +90deg (max).
    """
    value_clamped = float(np.clip(value, vmin, vmax))
    pct = (value_clamped - vmin) / (vmax - vmin + 1e-9)
    angle = -90 + (pct * 180)  # -90..+90

    st.markdown(
        f"""
        <div class="gauge-wrap">
          <div class="gauge">
            <div class="arc"></div>
            <div class="mask"></div>
            <div class="needle" style="transform: rotate({angle:.2f}deg);"></div>
            <div class="center"></div>
            <div class="label">{title}: {value_clamped:,.2f} {unit}</div>
            <div class="sub">{subtitle}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Header
# =========================
st.markdown(
    """
<div class="hero">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px;">
    <div>
      <div class="badge">☀️ SolarCast • Sri Lanka</div>
      <h1 style="margin:10px 0 6px 0;">Solar Forecasting Dashboard</h1>
      <div class="small">
        Predicts <b>ALLSKY_SFC_SW_DWN</b> (W/m²) and estimates PV energy (kWh) using <b>XGBoost</b>.
      </div>
    </div>
    <div style="text-align:right" class="tiny">
      Hourly 2020–2023 • 6 cities<br/>
      Output: Irradiance (W/m²) + PV kWh estimate
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.write("")

# =========================
# Validate required files
# =========================
if not Path(DATA_PATH).exists() or not Path(MODEL_PATH).exists():
    st.error(
        "Required files not found.\n\n"
        f"- {DATA_PATH}\n"
        f"- {MODEL_PATH}\n\n"
        "Run preprocessing and training first."
    )
    st.stop()

# =========================
# Load resources
# =========================
df = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

target = "ALLSKY_SFC_SW_DWN"
feature_cols = [c for c in df.columns if c not in ["timestamp", target]]
cities = sorted([c.replace("city_", "") for c in df.columns if c.startswith("city_")])

# Robust medians (fallbacks)
med_T2M = float(df["T2M"].median()) if "T2M" in df.columns else 28.0
med_RH2M = float(df["RH2M"].median()) if "RH2M" in df.columns else 75.0
med_WS2M = float(df["WS2M"].median()) if "WS2M" in df.columns else 2.0

# =========================
# Forecast Section (NO PLOTS)
# =========================
left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Forecast Inputs")

    c1, c2, c3 = st.columns(3)
    with c1:
        city = st.selectbox("City", cities, index=0 if cities else None)
    with c2:
        date = st.date_input("Date")
    with c3:
        hour = st.slider("Hour", 0, 23, 12)

    st.markdown("**Weather drivers**")
    w1, w2, w3 = st.columns(3)
    with w1:
        T2M = st.number_input("Temperature (°C)", value=med_T2M, step=0.5)
    with w2:
        RH2M = st.number_input("Humidity (%)", value=med_RH2M, step=1.0, min_value=0.0, max_value=100.0)
    with w3:
        WS2M = st.number_input("Wind Speed (m/s)", value=med_WS2M, step=0.2, min_value=0.0)

    st.markdown("<hr/>", unsafe_allow_html=True)

    st.markdown("**PV system settings (kWh estimate)**")
    p1, p2, p3 = st.columns(3)
    with p1:
        pv_area = st.number_input("Panel area (m²)", value=10.0, step=1.0, min_value=1.0)
    with p2:
        pv_eff = st.number_input("Efficiency (0–1)", value=0.18, step=0.01, min_value=0.05, max_value=0.30)
    with p3:
        pr = st.number_input("Performance ratio (0–1)", value=0.80, step=0.05, min_value=0.50, max_value=0.95)

    st.markdown("<hr/>", unsafe_allow_html=True)
    run = st.button("Run Forecast", type="primary", use_container_width=True)
    st.markdown(
        '<div class="tiny">Tip: Use realistic weather inputs. PV kWh is an estimate using area × efficiency × PR.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Results")

    if run:
        ts = pd.Timestamp(date.year, date.month, date.day, hour)

        # Build input row exactly like preprocessing expectations
        row = build_features_row(df, feature_cols, city, ts, T2M, RH2M, WS2M)
        X = pd.DataFrame([row], columns=feature_cols)

        # Predict irradiance
        irr = float(model.predict(X)[0])
        irr = max(0.0, irr)  # sanity

        st.caption(f"Forecast time: **{ts}** • City: **{city}**")

        # Irradiance gauge (needle animation) + level
        level = label_intensity(irr, max_irr=1000.0)
        render_gauge(
            title="Irradiance",
            value=irr,
            unit="W/m²",
            vmin=0.0,
            vmax=1000.0,
            subtitle=f"Intensity: {level}",
        )

        st.markdown("<hr/>", unsafe_allow_html=True)

        # PV energy estimate (1-hour)
        # PV Power (W) ≈ Irradiance (W/m²) * Area (m²) * Efficiency * PR
        pv_power_w = irr * float(pv_area) * float(pv_eff) * float(pr)
        pv_kwh_1h = pv_power_w / 1000.0  # 1 hour

        # PV meter with needle animation
        # Use a reasonable gauge max for UI (e.g., 0..5 kWh per hour) then auto-expand if needed
        pv_gauge_max = max(5.0, float(np.ceil(pv_kwh_1h * 1.2)))
        render_gauge(
            title="PV Energy (1 hour)",
            value=pv_kwh_1h,
            unit="kWh",
            vmin=0.0,
            vmax=pv_gauge_max,
            subtitle="Estimated from irradiance & PV settings",
        )

        st.markdown(
            f"""
            <div class="tiny" style="margin-top:8px;">
              <b>Note:</b> This PV value is an <i>estimate</i> (not measured generation). It assumes the selected PV area,
              module efficiency, and performance ratio (loss factor).
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:
        st.info("Enter inputs and click **Run Forecast** to see the animated gauges.")

    st.markdown("</div>", unsafe_allow_html=True)
