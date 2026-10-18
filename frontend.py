import json
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import altair as alt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components

# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000/predict"
CURRENT_YEAR = 2026

INK = "#12263A"      # deep navy: text, hero
TEAL = "#1B8A8F"     # primary accent
AMBER = "#E8A33D"    # "you are here" highlight
MIST = "#EDF2F6"     # page background
LINE = "#D5DEE6"     # borders and gridlines

# Button colors: change these two lines to recolor every button
BTN = "#5B4BDB"        # indigo
BTN_HOVER = "#4538B8"  # darker indigo for hover

NEIGHBORHOODS = {
    "CollgCr": "College Creek",
    "NAmes": "North Ames",
    "NridgHt": "Northridge Heights",
    "OldTown": "Old Town",
    "Edwards": "Edwards",
    "Other": "Other / not listed",
}

DEFAULTS = {
    "living_area": 1710,
    "overall_qual": 7,
    "bedrooms": 3,
    "bathrooms": 2,
    "year_built": 2003,
    "neighborhood": "CollgCr",
    "garage_cars": 2,
    "lot_area": 8450,
}

PRESETS = {
    "Starter home": dict(
        living_area=1100, overall_qual=5, bedrooms=2, bathrooms=1,
        year_built=1965, neighborhood="Edwards", garage_cars=1, lot_area=6500,
    ),
    "Family home": dict(
        living_area=1850, overall_qual=7, bedrooms=3, bathrooms=2,
        year_built=2003, neighborhood="CollgCr", garage_cars=2, lot_area=9000,
    ),
    "Luxury home": dict(
        living_area=3200, overall_qual=9, bedrooms=4, bathrooms=3,
        year_built=2008, neighborhood="NridgHt", garage_cars=3, lot_area=14000,
    ),
}

QUALITY_LABELS = {
    1: "Very poor", 2: "Poor", 3: "Fair", 4: "Below average", 5: "Average",
    6: "Above average", 7: "Good", 8: "Very good", 9: "Excellent", 10: "Luxury",
}

# Features the "Explore" tab can sweep: label -> (input key, values to try)
SWEEPS = {
    "Overall quality": ("overall_qual", list(range(1, 11))),
    "Living area (sq ft)": ("living_area", list(range(600, 4001, 200))),
    "Year built": ("year_built", list(range(1900, 2021, 10)) + [2026]),
    "Lot area (sq ft)": ("lot_area", list(range(2000, 30001, 2000))),
    "Garage capacity (cars)": ("garage_cars", [0, 1, 2, 3, 4]),
    "Bedrooms": ("bedrooms", list(range(1, 7))),
    "Full bathrooms": ("bathrooms", list(range(1, 5))),
}

# Price drivers: label, input key, step, min, max
DRIVERS = [
    ("Quality", "overall_qual", 2, 1, 10),
    ("Living area", "living_area", 300, 300, 10000),
    ("Year built", "year_built", 15, 1800, CURRENT_YEAR),
    ("Garage", "garage_cars", 1, 0, 4),
    ("Bedrooms", "bedrooms", 1, 0, 10),
    ("Bathrooms", "bathrooms", 1, 0, 6),
    ("Lot area", "lot_area", 2000, 500, 100000),
]

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide",
)

# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Manrope', sans-serif; }}
.stApp {{ background: {MIST}; color: {INK}; }}
.block-container {{ padding-top: 1.8rem; max-width: 1180px; }}
#MainMenu, footer {{ visibility: hidden; }}
h1, h2, h3, h4, label, p, span, div {{ color: {INK}; }}

.hero {{
    position: relative;
    overflow: hidden;
    background: linear-gradient(120deg, {INK} 0%, #1D4A5C 60%, {TEAL} 135%);
    border-radius: 22px;
    padding: 2.2rem 2.4rem;
    margin-bottom: 1.5rem;
}}
.hero h1 {{
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.4rem;
    margin: 0 0 .35rem 0;
    color: #fff;
}}
.hero p {{ color: #C9DAE4; margin: 0; font-size: 1.02rem; max-width: 540px; position: relative; z-index: 1; }}
.hero svg {{ position: absolute; right: 0; bottom: 0; height: 100%; opacity: .9; }}

.section-title {{
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 1.55rem;
    margin: 1.6rem 0 .2rem 0;
}}
.section-sub {{ color: #5D7487; margin-bottom: .6rem; }}

.panel {{
    background: #fff;
    border: 1px solid {LINE};
    border-radius: 16px;
    padding: 1.2rem 1.3rem;
}}
.panel-title {{ font-weight: 700; font-size: 1.05rem; margin-bottom: .6rem; }}

.summary-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: .7rem .9rem; }}
.stat .k {{ font-size: .78rem; color: #5D7487; }}
.stat .v {{ font-size: 1.1rem; font-weight: 700; }}
.hood {{
    display: inline-block;
    background: #E3F3F3;
    color: #0F6468;
    border-radius: 999px;
    padding: .2rem .8rem;
    font-weight: 600;
    font-size: .85rem;
    margin-bottom: .8rem;
}}

.stale {{
    background: #FFF4DE;
    border: 1px solid #F2D9A4;
    color: #7A5410;
    border-radius: 10px;
    padding: .5rem .8rem;
    font-size: .88rem;
    margin: .5rem 0;
}}

.takeaway {{
    background: #fff;
    border: 1px solid {LINE};
    border-left: 5px solid {TEAL};
    border-radius: 12px;
    padding: .7rem 1rem;
    margin: .6rem 0 .2rem 0;
}}

/* Secondary buttons: white with indigo hover (works even in a dark Streamlit theme) */
.stButton > button,
.stDownloadButton > button,
button[data-testid="stBaseButton-secondary"] {{
    background: #FFFFFF !important;
    color: {INK} !important;
    border: 1px solid {LINE} !important;
    border-radius: 10px;
    font-weight: 600;
    transition: transform .1s ease, border-color .15s ease, background .15s ease;
}}
.stButton > button:hover,
.stDownloadButton > button:hover,
button[data-testid="stBaseButton-secondary"]:hover {{
    background: #F0EEFF !important;
    border-color: {BTN} !important;
    color: {BTN} !important;
}}
.stButton > button:active {{ transform: scale(.98); }}

/* Primary button: solid indigo */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {{
    background: {BTN} !important;
    border: 1px solid {BTN} !important;
    color: #FFFFFF !important;
    padding: .75rem 1rem;
    font-size: 1.05rem;
}}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {{
    background: {BTN_HOVER} !important;
    border-color: {BTN_HOVER} !important;
    color: #FFFFFF !important;
}}

/* Text inside buttons follows the button color (the global text rule would make it dark) */
.stButton > button *, .stDownloadButton > button *,
button[data-testid^="stBaseButton"] * {{ color: inherit !important; }}

.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
.stTabs [aria-selected="true"] {{ color: {TEAL}; }}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# State
# --------------------------------------------------

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("result", None)
st.session_state.setdefault("history", [])
st.session_state.setdefault("insights", {})


def apply_preset(name: str) -> None:
    for key, value in PRESETS[name].items():
        st.session_state[key] = value


def randomize_inputs() -> None:
    """Fill every input with a plausible random house (values stay consistent with each other)."""
    quality = random.randint(3, 10)
    area = int(round(random.randint(700, 2200 + quality * 150) / 50) * 50)
    bedrooms = max(1, min(6, area // 600 + random.choice([-1, 0, 0, 1])))
    st.session_state.update(
        living_area=area,
        overall_qual=quality,
        bedrooms=bedrooms,
        bathrooms=max(1, min(4, bedrooms // 2 + random.choice([0, 1]))),
        year_built=random.randint(1925, 2010),
        neighborhood=random.choice(list(NEIGHBORHOODS)),
        garage_cars=random.choice([0, 1, 1, 2, 2, 2, 3]),
        lot_area=int(round(random.randint(4000, 16000) / 100) * 100),
    )


def reset_inputs() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


def current_inputs() -> dict:
    return {key: st.session_state[key] for key in DEFAULTS}


def signature(inputs: dict) -> str:
    return json.dumps(inputs, sort_keys=True)


# --------------------------------------------------
# Model API
# --------------------------------------------------

def build_features(inputs: dict) -> dict:
    """Match every feature and category to your training data."""
    living_area = inputs["living_area"]
    bedrooms = inputs["bedrooms"]
    bathrooms = inputs["bathrooms"]
    year_built = inputs["year_built"]
    garage_cars = inputs["garage_cars"]

    return {
        "MSSubClass": 20,
        "MSZoning": "RL",
        "LotArea": inputs["lot_area"],
        "Street": "Pave",
        "LotShape": "Reg",
        "LandContour": "Lvl",
        "Utilities": "AllPub",
        "LotConfig": "Inside",
        "LandSlope": "Gtl",
        "Neighborhood": inputs["neighborhood"],
        "Condition1": "Norm",
        "Condition2": "Norm",
        "BldgType": "1Fam",
        "HouseStyle": "2Story",
        "OverallQual": inputs["overall_qual"],
        "OverallCond": 5,
        "YearBuilt": year_built,
        "YearRemodAdd": year_built,
        "RoofStyle": "Gable",
        "RoofMatl": "CompShg",
        "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd",
        "MasVnrArea": 0,
        "ExterQual": "Gd",
        "ExterCond": "TA",
        "Foundation": "PConc",
        "BsmtQual": "Gd",
        "BsmtCond": "TA",
        "BsmtExposure": "No",
        "BsmtFinType1": "GLQ",
        "BsmtFinSF1": 706,
        "BsmtFinType2": "Unf",
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 150,
        "TotalBsmtSF": 856,
        "Heating": "GasA",
        "HeatingQC": "Ex",
        "CentralAir": "Y",
        "Electrical": "SBrkr",
        "1stFlrSF": 856,
        "2ndFlrSF": max(0, living_area - 856),
        "GrLivArea": living_area,
        "BsmtFullBath": 1,
        "BsmtHalfBath": 0,
        "FullBath": bathrooms,
        "HalfBath": 0,
        "BedroomAbvGr": bedrooms,
        "KitchenAbvGr": 1,
        "KitchenQual": "Gd",
        "TotRmsAbvGrd": bedrooms + bathrooms + 2,
        "Functional": "Typ",
        "Fireplaces": 0,
        "GarageType": "Attchd" if garage_cars > 0 else "Detchd",
        "GarageYrBlt": year_built if garage_cars > 0 else None,
        "GarageFinish": "RFn" if garage_cars > 0 else "Missing",
        "GarageCars": garage_cars,
        "GarageArea": garage_cars * 250,
        "GarageQual": "TA" if garage_cars > 0 else "Missing",
        "GarageCond": "TA" if garage_cars > 0 else "Missing",
        "PavedDrive": "Y",
        "WoodDeckSF": 0,
        "OpenPorchSF": 61,
        "EnclosedPorch": 0,
        "ScreenPorch": 0,
        "MoSold": 6,
        "YrSold": 2008,
        "SaleType": "WD",
        "SaleCondition": "Normal",
    }


def fetch_price(inputs: dict) -> float:
    response = requests.post(
        API_URL,
        json={"features": build_features(inputs)},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"API error ({response.status_code}): {response.text}")
    return float(response.json()["predicted_price"])


# Shared cache for the analysis charts (they re-ask the model for many
# variations of the same house). Cleared with the "Clear cached analysis" button.
_PRICE_CACHE: dict = {}


def cached_price(inputs: dict) -> float:
    key = signature(inputs)
    if key not in _PRICE_CACHE:
        _PRICE_CACHE[key] = fetch_price(inputs)
    return _PRICE_CACHE[key]


def run_batch(variants: list, message: str):
    """Query the model for many variations in parallel. Returns None on failure."""
    try:
        with st.spinner(message):
            with ThreadPoolExecutor(max_workers=8) as pool:
                return list(pool.map(cached_price, variants))
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI. Start the backend first.")
    except requests.exceptions.RequestException as exc:
        st.error(f"Request failed: {exc}")
    except RuntimeError as exc:
        st.error(str(exc))
    except (KeyError, ValueError) as exc:
        st.error(f"Invalid API response: {exc}")
    return None


# --------------------------------------------------
# Chart helpers
# --------------------------------------------------

def style_chart(chart):
    return (
        chart.configure_axis(
            labelColor=INK, titleColor=INK, gridColor="#E6ECF1", domainColor=LINE,
            labelFont="Manrope", titleFont="Manrope",
        )
        .configure_view(strokeWidth=0)
        .configure(background="#FFFFFF")
    )


def money_axis(title="Estimated price"):
    return alt.Axis(format="$,.0f", title=title)


def fmt_num(value) -> str:
    return f"{value:,}"


# --------------------------------------------------
# Animated price card (count-up and gauge)
# --------------------------------------------------

CARD_HTML = """
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700&family=Manrope:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  body { margin: 0; font-family: 'Manrope', sans-serif; }
  .card { display: flex; flex-wrap: wrap; align-items: center; gap: 16px;
          background: #fff; border: 1px solid #D5DEE6; border-left: 6px solid #E8A33D;
          border-radius: 16px; padding: 18px 22px; box-sizing: border-box; min-height: 196px; }
  .left { flex: 1 1 210px; min-width: 0; }
  .label { color: #5D7487; font-size: 14px; }
  .price { font-family: 'Fraunces', serif; font-weight: 700; font-size: 44px; color: #12263A; line-height: 1.1; }
  .sub { color: #4A6275; font-size: 14px; margin-top: 4px; }
  .up { color: #1A7F4B; font-weight: 700; } .down { color: #B3402F; font-weight: 700; }
  .gauge { flex: 0 0 190px; text-align: center; }
  .tier { font-weight: 700; font-size: 14px; fill: #12263A; }
  .small { font-size: 11px; fill: #5D7487; }
</style>
<div class="card">
  <div class="left">
    <div class="label">Estimated sale price</div>
    <div class="price" id="price">$0</div>
    <div class="sub">$__PERSQFT__ per sq ft__DELTA__</div>
    <div class="sub">Rough range: $__LOW__ to $__HIGH__</div>
  </div>
  <div class="gauge">
    <svg viewBox="0 0 240 150" width="190">
      <defs>
        <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stop-color="#BFE3E3"/><stop offset=".55" stop-color="#1B8A8F"/><stop offset="1" stop-color="#12263A"/>
        </linearGradient>
      </defs>
      <path d="M 20 120 A 100 100 0 0 1 220 120" fill="none" stroke="#E6ECF1" stroke-width="16" stroke-linecap="round"/>
      <path id="arc" d="M 20 120 A 100 100 0 0 1 220 120" fill="none" stroke="url(#g)" stroke-width="16"
            stroke-linecap="round" stroke-dasharray="314.16" stroke-dashoffset="314.16"/>
      <circle id="dot" cx="20" cy="120" r="9" fill="#E8A33D" stroke="#fff" stroke-width="3"/>
      <text id="tier" class="tier" x="120" y="108" text-anchor="middle">Mid-range</text>
      <text class="small" x="20" y="144" text-anchor="middle">$50k</text>
      <text class="small" x="220" y="144" text-anchor="middle">$500k+</text>
    </svg>
  </div>
</div>
<script>
  const target = __PRICE__, start = __START__, lo = 50000, hi = 500000;
  const fmt = n => '$' + Math.round(n).toLocaleString('en-US');
  const frac = p => Math.min(Math.max((p - lo) / (hi - lo), 0.02), 0.98);
  const tier = p => p < 140000 ? 'Entry-level' : p < 240000 ? 'Mid-range' : p < 380000 ? 'Upper-tier' : 'Premium';
  const priceEl = document.getElementById('price'), arc = document.getElementById('arc'),
        dot = document.getElementById('dot'), tierEl = document.getElementById('tier');
  const t0 = performance.now(), dur = 1100;
  function tick(t) {
    let k = Math.min((t - t0) / dur, 1); k = 1 - Math.pow(1 - k, 3);
    const p = start + (target - start) * k, f = frac(p), th = Math.PI * (1 - f);
    priceEl.textContent = fmt(p);
    arc.style.strokeDashoffset = 314.16 * (1 - f);
    dot.setAttribute('cx', 120 + 100 * Math.cos(th));
    dot.setAttribute('cy', 120 - 100 * Math.sin(th));
    tierEl.textContent = tier(p);
    if (k < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
</script>
"""


def render_price_card(result: dict) -> None:
    price = result["price"]
    previous = result["previous"]
    per_sqft = price / result["inputs"]["living_area"]

    delta = ""
    if previous and abs(price - previous) >= 1:
        diff = price - previous
        cls, arrow = ("up", "▲") if diff > 0 else ("down", "▼")
        delta = f' · <span class="{cls}">{arrow} ${abs(diff):,.0f}</span>'

    html = (
        CARD_HTML.replace("__PRICE__", f"{price:.2f}")
        .replace("__START__", f"{previous or 0:.2f}")
        .replace("__PERSQFT__", f"{per_sqft:,.0f}")
        .replace("__DELTA__", delta)
        .replace("__LOW__", f"{price * 0.92:,.0f}")
        .replace("__HIGH__", f"{price * 1.08:,.0f}")
    )
    components.html(html, height=215)


# --------------------------------------------------
# Hero
# --------------------------------------------------

st.markdown(
    """
<div class="hero">
  <h1>🏡 House Price Predictor</h1>
  <p>Describe a property and get an instant estimate. Then use the charts below to see what drives the price and how it compares.</p>
  <svg viewBox="0 0 420 200" preserveAspectRatio="xMaxYMax meet" xmlns="http://www.w3.org/2000/svg">
    <g fill="rgba(255,255,255,0.07)">
      <rect x="20" y="110" width="50" height="90"/><rect x="80" y="70" width="40" height="130"/>
      <rect x="130" y="130" width="60" height="70"/><polygon points="200,200 200,120 245,80 290,120 290,200"/>
      <rect x="300" y="90" width="45" height="110"/><rect x="355" y="140" width="55" height="60"/>
    </g>
    <g fill="rgba(232,163,61,0.55)">
      <rect x="236" y="140" width="18" height="60"/>
    </g>
  </svg>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Inputs (left) and live summary + result (right)
# --------------------------------------------------

left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="panel-title">Start from a preset</div>', unsafe_allow_html=True)
    p1, p2, p3, p4, p5 = st.columns(5)
    for col, name in zip((p1, p2, p3), PRESETS):
        col.button(name, on_click=apply_preset, args=(name,), use_container_width=True)
    p4.button("🎲 Random", on_click=randomize_inputs, use_container_width=True)
    p5.button("Reset", on_click=reset_inputs, use_container_width=True)

    st.write("")
    tab_size, tab_quality, tab_location = st.tabs(
        ["📐 Size & layout", "⭐ Quality & age", "📍 Location & extras"]
    )

    with tab_size:
        st.number_input("Living area (sq. ft.)", 300, 10000, step=50, key="living_area")
        c1, c2 = st.columns(2)
        c1.number_input("Bedrooms above ground", 0, 10, key="bedrooms")
        c2.number_input("Full bathrooms", 0, 6, key="bathrooms")
        st.number_input("Lot area (sq. ft.)", 500, 100000, step=100, key="lot_area")

    with tab_quality:
        st.slider("Overall quality", 1, 10, key="overall_qual")
        st.caption(f"{st.session_state.overall_qual}/10: {QUALITY_LABELS[st.session_state.overall_qual]}")
        st.number_input("Year built", 1800, CURRENT_YEAR, key="year_built")

    with tab_location:
        st.selectbox(
            "Neighborhood",
            options=list(NEIGHBORHOODS),
            format_func=lambda code: f"{NEIGHBORHOODS[code]} ({code})",
            key="neighborhood",
        )
        st.radio("Garage capacity (cars)", [0, 1, 2, 3, 4], horizontal=True, key="garage_cars")

inputs = current_inputs()
sig = signature(inputs)

with right:
    age = CURRENT_YEAR - inputs["year_built"]
    garage_text = "No garage" if inputs["garage_cars"] == 0 else f"{inputs['garage_cars']}-car"
    st.markdown(
        f"""
<div class="panel">
  <div class="panel-title">Your property</div>
  <span class="hood">{NEIGHBORHOODS[inputs['neighborhood']]}</span>
  <div class="summary-grid">
    <div class="stat"><div class="k">Living area</div><div class="v">{inputs['living_area']:,} sq ft</div></div>
    <div class="stat"><div class="k">Lot</div><div class="v">{inputs['lot_area']:,} sq ft</div></div>
    <div class="stat"><div class="k">Bedrooms / baths</div><div class="v">{inputs['bedrooms']} bd / {inputs['bathrooms']} ba</div></div>
    <div class="stat"><div class="k">Quality</div><div class="v">{inputs['overall_qual']}/10 · {QUALITY_LABELS[inputs['overall_qual']]}</div></div>
    <div class="stat"><div class="k">Built</div><div class="v">{inputs['year_built']} ({age} yrs old)</div></div>
    <div class="stat"><div class="k">Garage</div><div class="v">{garage_text}</div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    auto = st.toggle("Update the estimate as I change inputs", key="auto_update")
    predict_clicked = st.button("Predict house price", type="primary", use_container_width=True)

    result = st.session_state.result
    needs_run = predict_clicked or (auto and (result is None or result["inputs"] != inputs))

    if needs_run:
        try:
            with st.spinner("Predicting house price..."):
                price = fetch_price(inputs)

            previous = result["price"] if result else None
            st.session_state.result = {"price": price, "inputs": dict(inputs), "previous": previous}

            if predict_clicked:
                st.session_state.history.append(
                    {
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Area (sq ft)": inputs["living_area"],
                        "Quality": inputs["overall_qual"],
                        "Beds": inputs["bedrooms"],
                        "Baths": inputs["bathrooms"],
                        "Built": inputs["year_built"],
                        "Neighborhood": NEIGHBORHOODS[inputs["neighborhood"]],
                        "Garage": inputs["garage_cars"],
                        "Price ($)": round(price, 2),
                    }
                )
                st.toast("Estimate ready", icon="🏡")

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI. Start the backend first.")
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
        except RuntimeError as exc:
            st.error(str(exc))
        except (KeyError, ValueError) as exc:
            st.error(f"Invalid API response: {exc}")

    result = st.session_state.result
    if result:
        render_price_card(result)
        st.caption(
            "This is a model estimate, not a guaranteed sale price. "
            "The range is a simple ±8% band, not a statistical confidence interval."
        )
        if result["inputs"] != inputs:
            st.markdown(
                '<div class="stale">You changed some details since this estimate. '
                "Press <b>Predict house price</b> to update it.</div>",
                unsafe_allow_html=True,
            )

# --------------------------------------------------
# Insights
# --------------------------------------------------

st.markdown('<div class="section-title">Insights</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-sub">Each chart asks your model about variations of the property above. '
    "The amber marker always shows where your current choice sits.</div>",
    unsafe_allow_html=True,
)

if st.button("Clear cached analysis"):
    _PRICE_CACHE.clear()
    st.session_state.insights = {}

tab_drivers, tab_explore, tab_hoods, tab_heat, tab_history = st.tabs(
    ["📊 Price drivers", "📈 Explore a feature", "🏘️ Neighborhoods", "🔥 Quality × size", "🕘 History"]
)


def stale_note(key: str) -> None:
    saved = st.session_state.insights.get(key)
    if saved and saved["sig"] != sig:
        st.markdown(
            '<div class="stale">These results are for earlier inputs. Run the analysis again to refresh them.</div>',
            unsafe_allow_html=True,
        )


# ---------- Price drivers (tornado) ----------
with tab_drivers:
    st.write("What happens to the estimate if one thing changes and everything else stays the same?")
    if st.button("Analyze price drivers", key="run_drivers"):
        variants, meta = [], []
        for name, key, step, lo, hi in DRIVERS:
            for direction in (-1, 1):
                new_value = min(max(inputs[key] + direction * step, lo), hi)
                if new_value != inputs[key]:
                    variants.append({**inputs, key: new_value})
                    meta.append(f"{name} {fmt_num(inputs[key])} → {fmt_num(new_value)}")
        prices = run_batch([inputs] + variants, "Testing one change at a time...")
        if prices:
            base, others = prices[0], prices[1:]
            df = pd.DataFrame({"Change": meta, "Delta": [p - base for p in others]})
            st.session_state.insights["drivers"] = {"sig": sig, "df": df, "base": base}

    saved = st.session_state.insights.get("drivers")
    if saved:
        stale_note("drivers")
        df = saved["df"]
        order = df.reindex(df["Delta"].abs().sort_values(ascending=False).index)["Change"].tolist()
        base = alt.Chart(df).encode(
            y=alt.Y("Change:N", sort=order, title=None),
            x=alt.X("Delta:Q", title="Change in estimated price ($)", axis=alt.Axis(format="+$,.0f")),
        )
        bars = base.mark_bar(cornerRadius=4, height=22).encode(
            color=alt.condition(alt.datum.Delta > 0, alt.value(TEAL), alt.value(AMBER)),
            tooltip=[alt.Tooltip("Change:N"), alt.Tooltip("Delta:Q", format="+$,.0f")],
        )
        labels_pos = base.transform_filter(alt.datum.Delta > 0).mark_text(align="left", dx=6, color=INK).encode(
            text=alt.Text("Delta:Q", format="+$,.0f"))
        labels_neg = base.transform_filter(alt.datum.Delta <= 0).mark_text(align="right", dx=-6, color=INK).encode(
            text=alt.Text("Delta:Q", format="+$,.0f"))
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK).encode(x="x:Q")
        st.altair_chart(
            style_chart(alt.layer(bars, labels_pos, labels_neg, zero).properties(height=34 * len(df) + 20)),
            use_container_width=True,
        )
        top = df.loc[df["Delta"].abs().idxmax()]
        st.markdown(
            f'<div class="takeaway">Biggest swing: <b>{top["Change"]}</b> moves the estimate by '
            f'<b>{top["Delta"]:+,.0f}</b> dollars.</div>',
            unsafe_allow_html=True,
        )

# ---------- Explore a feature (hover curve) ----------
with tab_explore:
    feature = st.selectbox("Feature to sweep", list(SWEEPS), key="explore_feature")
    if st.button("Plot price curve", key="run_explore"):
        key, values = SWEEPS[feature]
        prices = run_batch([{**inputs, key: v} for v in values], f"Sweeping {feature.lower()}...")
        if prices:
            st.session_state.insights["explore"] = {
                "sig": sig, "feature": feature, "current": inputs[key],
                "df": pd.DataFrame({"Value": values, "Price": prices}),
            }

    saved = st.session_state.insights.get("explore")
    if saved:
        stale_note("explore")
        df, name = saved["df"], saved["feature"]
        base = alt.Chart(df).encode(
            x=alt.X("Value:Q", title=name, scale=alt.Scale(zero=False)),
            y=alt.Y("Price:Q", axis=money_axis(), scale=alt.Scale(zero=False)),
        )
        area = base.mark_area(
            opacity=0.9,
            line=False,
            color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color="#FFFFFF", offset=0), alt.GradientStop(color="#BFE3E3", offset=1)],
                x1=1, x2=1, y1=1, y2=0,
            ),
        )
        line = base.mark_line(color=TEAL, strokeWidth=3, interpolate="monotone")
        nearest = alt.selection_point(nearest=True, on="mouseover", fields=["Value"], empty=False)
        catcher = base.mark_point().encode(opacity=alt.value(0)).add_params(nearest)
        points = base.mark_circle(color=TEAL, size=120).encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
        rule = base.mark_rule(color="#9FB3C2").encode(
            tooltip=[alt.Tooltip("Value:Q", title=name), alt.Tooltip("Price:Q", format="$,.0f")]
        ).transform_filter(nearest)
        current = alt.Chart(pd.DataFrame({"Value": [saved["current"]]})).mark_rule(
            color=AMBER, strokeDash=[6, 4], strokeWidth=2).encode(x="Value:Q")
        st.altair_chart(
            style_chart(alt.layer(area, line, catcher, points, rule, current).properties(height=330)),
            use_container_width=True,
        )
        st.caption("Hover the curve to read exact values. The dashed amber line is your current setting.")

# ---------- Neighborhoods ----------
with tab_hoods:
    st.write("Same house, different neighborhood.")
    if st.button("Compare neighborhoods", key="run_hoods"):
        codes = list(NEIGHBORHOODS)
        prices = run_batch([{**inputs, "neighborhood": c} for c in codes], "Comparing neighborhoods...")
        if prices:
            df = pd.DataFrame({
                "Code": codes, "Neighborhood": [NEIGHBORHOODS[c] for c in codes], "Price": prices,
            })
            st.session_state.insights["hoods"] = {"sig": sig, "df": df, "current": inputs["neighborhood"]}

    saved = st.session_state.insights.get("hoods")
    if saved:
        stale_note("hoods")
        df = saved["df"]
        order = df.sort_values("Price", ascending=False)["Neighborhood"].tolist()
        base = alt.Chart(df).encode(
            y=alt.Y("Neighborhood:N", sort=order, title=None),
            x=alt.X("Price:Q", axis=money_axis(), scale=alt.Scale(zero=True)),
        )
        bars = base.mark_bar(cornerRadiusEnd=5, height=26).encode(
            color=alt.condition(alt.datum.Code == saved["current"], alt.value(AMBER), alt.value(TEAL)),
            tooltip=[alt.Tooltip("Neighborhood:N"), alt.Tooltip("Price:Q", format="$,.0f")],
        )
        text = base.mark_text(align="left", dx=6, color=INK).encode(text=alt.Text("Price:Q", format="$,.0f"))
        st.altair_chart(style_chart(alt.layer(bars, text).properties(height=48 * len(df))),
                        use_container_width=True)
        hi, lo = df.loc[df["Price"].idxmax()], df.loc[df["Price"].idxmin()]
        st.markdown(
            f'<div class="takeaway">For this house, <b>{hi["Neighborhood"]}</b> is highest and '
            f'<b>{lo["Neighborhood"]}</b> is lowest, a gap of <b>${hi["Price"] - lo["Price"]:,.0f}</b>.</div>',
            unsafe_allow_html=True,
        )

# ---------- Heatmap: quality x size ----------
with tab_heat:
    st.write("How quality and living area combine. Darker cells mean higher prices.")
    if st.button("Build heatmap", key="run_heat"):
        qualities = list(range(3, 11))
        areas = sorted(set([800, 1200, 1600, 2000, 2400, 2800, 3200] + [inputs["living_area"]]))
        grid = [(q, a) for q in qualities for a in areas]
        prices = run_batch(
            [{**inputs, "overall_qual": q, "living_area": a} for q, a in grid],
            f"Testing {len(grid)} combinations...",
        )
        if prices:
            df = pd.DataFrame(grid, columns=["Quality", "Area"])
            df["Price"] = prices
            df["Label"] = df["Price"].apply(lambda p: f"${p / 1000:,.0f}k")
            df["Area"] = df["Area"].astype(str)
            st.session_state.insights["heat"] = {
                "sig": sig, "df": df,
                "q": inputs["overall_qual"], "a": str(inputs["living_area"]),
                "area_order": [str(a) for a in areas],
            }

    saved = st.session_state.insights.get("heat")
    if saved:
        stale_note("heat")
        df = saved["df"]
        mid = float(df["Price"].median())
        base = alt.Chart(df).encode(
            x=alt.X("Area:O", sort=saved["area_order"], title="Living area (sq ft)"),
            y=alt.Y("Quality:O", sort="descending", title="Overall quality"),
        )
        rects = base.mark_rect(cornerRadius=4).encode(
            color=alt.Color("Price:Q", scale=alt.Scale(range=["#E3F3F3", TEAL, INK]), legend=None),
            tooltip=["Quality:O", "Area:O", alt.Tooltip("Price:Q", format="$,.0f")],
        )
        text = base.mark_text(fontSize=12, fontWeight=600).encode(
            text="Label:N",
            color=alt.condition(alt.datum.Price > mid, alt.value("white"), alt.value(INK)),
        )
        here = alt.Chart(df[(df["Quality"] == saved["q"]) & (df["Area"] == saved["a"])]).mark_rect(
            fill=None, stroke=AMBER, strokeWidth=3).encode(
            x=alt.X("Area:O", sort=saved["area_order"]), y=alt.Y("Quality:O", sort="descending"))
        st.altair_chart(style_chart(alt.layer(rects, text, here).properties(height=380)),
                        use_container_width=True)
        st.caption("The amber outline is your current quality and living area.")

# ---------- History ----------
with tab_history:
    history = st.session_state.history
    if not history:
        st.info("No estimates yet. Press Predict house price to start a history.")
    else:
        hist_df = pd.DataFrame(history)
        hist_df["Estimate"] = range(1, len(hist_df) + 1)
        st.dataframe(
            hist_df.drop(columns="Estimate").iloc[::-1],
            use_container_width=True,
            hide_index=True,
            column_config={"Price ($)": st.column_config.NumberColumn(format="$%,.0f")},
        )
        if len(hist_df) > 1:
            trend = (
                alt.Chart(hist_df)
                .mark_line(color=TEAL, strokeWidth=3, point=alt.OverlayMarkDef(color=AMBER, size=90))
                .encode(
                    x=alt.X("Estimate:O", title="Estimate #"),
                    y=alt.Y("Price ($):Q", axis=money_axis(), scale=alt.Scale(zero=False)),
                    tooltip=["Neighborhood", "Area (sq ft)", "Quality", alt.Tooltip("Price ($):Q", format="$,.0f")],
                )
                .properties(height=260)
            )
            st.altair_chart(style_chart(trend), use_container_width=True)

        c1, c2 = st.columns(2)
        c1.download_button(
            "Download history (CSV)",
            hist_df.drop(columns="Estimate").to_csv(index=False),
            file_name="price_estimates.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if c2.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.result = None
            st.rerun()