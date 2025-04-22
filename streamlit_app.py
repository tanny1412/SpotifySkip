"""
Streamlit UI to interact with the Spotify Skip Prediction FastAPI.
"""
import streamlit as st
import pandas as pd
import requests
import pickle
from sklearn.preprocessing import LabelEncoder
from PIL import Image

# Load local model
@st.cache_resource
def load_local_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

# Page config and custom styling
st.set_page_config(page_title="Spotify Skip Predictor", page_icon="🎵", layout="wide")
st.markdown(
    """
    <style>
        html, body, [class*="stApp"], .block-container {
            background-color: #191414;
            color: #FFFFFF;
            font-family: 'Circular', 'Inter', 'Helvetica Neue', Arial, sans-serif;
        }
        section[data-testid="stSidebar"] {
            background-color: #191414;
        }
        section[data-testid="stSidebar"] p {
            color: #B3B3B3 !important;
        }
        .stButton>button {
            background-color: #1DB954 !important;
            color: #FFFFFF !important;
            border: none;
            border-radius: 8px !important;
            padding: 0.5em 1em !important;
            transition: opacity 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton>button:hover {
            opacity: 0.8 !important;
            box-shadow: 0 0 8px #1DB954 !important;
        }
        .logo-img:hover {
            filter: grayscale(100%);
            opacity: 0.8;
            transform: scale(1.05);
            transition: all 0.2s ease-in-out;
        }
        .result-box {
            animation: fadeIn 0.5s ease-in-out;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        /* Make only Predict Skip button bigger */
        .custom-predict-button button {
            font-size: 18px !important;
            padding: 0.75em 2.5em !important;
            box-shadow: 0 0 10px #1DB95480;
            transition: transform 0.2s ease;
        }

        .custom-predict-button button:hover {
            transform: scale(1.05);
            box-shadow: 0 0 14px #1DB954;
        }

        input[type="range"] {
            accent-color: #1DB954;
        }
        select {
            background-color: #121212;
            color: #FFFFFF;
            padding: 0.25em;
            border-radius: 4px;
            border: 1px solid #333333;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title & Team info
st.markdown(
    """<h1 style='color:#FFFFFF; margin:0; font-weight:bold;'>Spotify Skip Predictor</h1>""",
    unsafe_allow_html=True,
)
st.markdown(
    """<p style='color:#F2F2F2; font-size:16px; margin-top:-8px;'>Predict if a user will skip a track!</p>""",
    unsafe_allow_html=True,
)
st.markdown(
    """<div style='background-color:#1DB954; padding:10px; border-radius:8px;'>
      <h2 style='color:#ffffff; margin:0;'>Worcester Polytechnic Institute</h2>
      <p style='color:#ffffff; margin:0; font-size:14px;'><strong>Team 11</strong>: Tanish, Abeer, Manadar, Anurag</p>
    </div>""",
    unsafe_allow_html=True,
)

# Load data
@st.cache_data
def load_raw_data():
    return pd.read_csv("spotify_history.csv", encoding="UTF-8-SIG")

df_raw = load_raw_data()
encoder_p = LabelEncoder().fit(df_raw['platform'])
encoder_rs = LabelEncoder().fit(df_raw['reason_start'])

# Sidebar Inputs
st.sidebar.header("🎛️ Track Playback Features")
st.sidebar.markdown("### **⏱️ Time**")
hour = st.sidebar.slider(
    "⏰ Hour of Day (0–23)", 0, 23, 12,
    help="Hour of day when playback started (0 = midnight, 23 = 11 PM)"
)
st.sidebar.caption("Hour of day when playback started (0 = midnight, 23 = 11 PM)")

month = st.sidebar.slider(
    "📆 Month (1–12)", 1, 12, 1,
    help="Month of year (to detect seasonal patterns)"
)
st.sidebar.caption("Month of year (to detect seasonal patterns)")


st.sidebar.markdown("### **📅 Context**")
weekday_names = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weekday_sel = st.sidebar.selectbox(
    "📅 Day of Week", weekday_names,
    help="Weekday context — e.g. Monday morning vs. weekend evenings"
)
st.sidebar.caption("Day of week when playback started")
weekday = weekday_names.index(weekday_sel)

st.sidebar.markdown("### **💻 Device**")
platform = st.sidebar.selectbox("📱 Platform", encoder_p.classes_)
platform_descriptions = {
    'web player': 'Spotify Web Player in browser',
    'windows': 'Spotify Desktop App on Windows',
    'android': 'Android Spotify App',
    'iOS': 'iOS Spotify App',
    'cast to device': 'Casting via Chromecast or smart speaker',
    'mac': 'Spotify Desktop App on macOS',
}
st.sidebar.caption(f"{platform_descriptions.get(platform, 'Unknown')}")

st.sidebar.markdown("### **▶️ Playback**")
reason_start = st.sidebar.selectbox("▶️ Reason Start", encoder_rs.classes_)
reason_start_descriptions = {
    'autoplay': 'Started by Spotify’s autoplay (recommended next track)',
    'clickrow': 'User clicked the track in the playlist or library',
    'trackdone': 'Auto-start after previous track finished',
    'nextbtn': 'User pressed the \"Next\" button',
    'backbtn': 'User pressed the \"Back\" button',
    'unknown': 'No recorded reason',
    'popup': 'Started via a popup (e.g., queue or search result)',
    'appload': 'App loaded and track auto-started',
    'fwdbtn': 'User pressed the \"Forward\" button',
    'trackerror': 'Playback started after track error fallback',
    'remote': 'Started from a remote device or controller',
    'endplay': 'Next track auto-started when previous ended',
    'playbtn': 'User pressed the \"Play\" button',
}
st.sidebar.caption(f"{reason_start_descriptions.get(reason_start, 'Unknown')}")

shuffle = st.sidebar.checkbox("🔁 Shuffle Mode", False)

# Logo in center
st.markdown("<div style='margin-top:30px'></div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="spotify-logo">', unsafe_allow_html=True)
    st.image("image.png", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Powered by Spotify®")

# Payload
payload = {
    "hour": hour,
    "month": month,
    "weekday": weekday,
    "platform": int(encoder_p.transform([platform])[0]),
    "reason_start": int(encoder_rs.transform([reason_start])[0]),
    "shuffle": 1 if shuffle else 0,
}

# Centered, enlarged Predict Skip button
st.markdown("<div style='margin-top:20px'></div>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([2, 2, 1])
with col2:
    st.markdown('<div class="custom-predict-button">', unsafe_allow_html=True)
    predict_clicked = st.button("Predict Skip 🚀")
    st.markdown('</div>', unsafe_allow_html=True)

if predict_clicked:
    try:
        response = requests.post("http://localhost:8000/predict", json=payload)
        response.raise_for_status()
        result = response.json()
        prob = result['probability']
        source = "API"
    except Exception:
        st.info("API not reachable, using local model...")
        model = load_local_model()
        scaler = model.named_steps.get('scaler') if hasattr(model, 'named_steps') else None
        cols = list(scaler.feature_names_in_) if scaler and hasattr(scaler, 'feature_names_in_') else \
               ['platform', 'reason_start', 'shuffle', 'hour', 'month', 'weekday']
        df_input = pd.DataFrame([{c: payload[c] for c in cols}], columns=cols)
        prob = model.predict_proba(df_input)[0][1]
        source = "local model"

    prob_pct = int(round(prob * 100))
    if prob > 0.8:
        emoji, message, bg_color = "⏭️", f"High chance of skip ({prob_pct}%)", "#FFA500"
    elif prob < 0.2:
        emoji, message, bg_color = "✅", f"Very likely to listen through ({100 - prob_pct}%)", "#1DB954"
    else:
        emoji, message, bg_color = "🤔", f"Skip probability {prob_pct}%", "#2F3F4F"

    card_html = f"""
    <div class="result-box" style="background-color:{bg_color}; border-radius:8px; padding:16px; margin-top:20px">
        <h3 style="margin:0; color:#FFFFFF">{emoji} {message} [{source}]</h3>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
    st.progress(prob)
