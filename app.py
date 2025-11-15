# app.py
"""
Production-ready Streamlit Air Quality Dashboard with improved SMS alerting.
- Improved SMS logic: weighted AQI, nearest-3 stations, trend detection, threshold-based alerts,
  rate-limiting, robust Twilio validation and helpful error messages.
- Safe fallbacks for optional dependencies (geopandas, kriging module).
- Reads sensitive keys from environment variables (recommended for Streamlit Cloud).
"""

import os
import time
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import requests
import streamlit as st

# Optional imports (not required to run everything)
try:
    import geopandas as gpd
except Exception:
    gpd = None

# optional kriging module (your existing module)
try:
    from krigging import perform_kriging_correct
except Exception:
    perform_kriging_correct = None

# -------------
# CONFIG / SECRETS
# -------------
# IMPORTANT: For production, set these in Streamlit Cloud or your environment variables.
WAQI_API_TOKEN = os.environ.get("WAQI_API_TOKEN", "")  # e.g., "97a0e7..."
OPEN_METEO_BASE = os.environ.get("OPEN_METEO_BASE", "https://api.open-meteo.com")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.environ.get("TWILIO_PHONE_NUMBER", "")  # e.g., +1xxxx

# Default bounding box and center for Delhi
DELHI_BOUNDS = "28.404,76.840,28.883,77.349"
DELHI_LAT = 28.6139
DELHI_LON = 77.2090
DELHI_GEOJSON_URL = "https://raw.githubusercontent.com/udit-001/india-maps-data/master/geojson/delhi.geojson"

# Rate-limiting period (seconds)
SMS_RATE_LIMIT_SECONDS = 3 * 3600  # 3 hours

# Trend sensitivity for AQI changes
TREND_DELTA = 8  # AQI points difference to consider rising/falling

# Streamlit page config
st.set_page_config(page_title="Delhi Air Quality Dashboard", layout="wide", page_icon="💨")

# CSS - minimal to keep look good (you can paste your fancy CSS here)
st.markdown(
    """
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #0D47A1; text-align:center; }
    .subtitle { text-align:center; color:#1565C0; margin-bottom:1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------
# Utility helpers
# ------------------------
def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometers."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def safe_to_number(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def get_aqi_category(aqi):
    """Return (category, color_rgb, emoji, advice)"""
    try:
        aqi = float(aqi)
    except Exception:
        return "Unknown", [128, 128, 128], "❓", "No data"

    if aqi <= 50:
        return "Good", [0, 158, 96], "✅", "Enjoy outdoor activities."
    if aqi <= 100:
        return "Moderate", [255, 214, 0], "🟡", "Unusually sensitive people should consider reducing prolonged exertion."
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups", [249, 115, 22], "🟠", "Sensitive groups should reduce prolonged exertion."
    if aqi <= 200:
        return "Unhealthy", [220, 38, 38], "🔴", "Everyone may begin to experience health effects."
    if aqi <= 300:
        return "Very Unhealthy", [147, 51, 234], "🟣", "Health alert: everyone may experience more serious effects."
    return "Hazardous", [126, 34, 206], "☠️", "Emergency conditions. Avoid all outdoor activity."


# ------------------------
# Caching network calls
# ------------------------
@st.cache_data(ttl=600)
def fetch_live_aqi(delhi_bounds=DELHI_BOUNDS, token=WAQI_API_TOKEN):
    """Fetch live AQI stations from WAQI map/bounds endpoint."""
    if not token:
        st.warning("WAQI API token not configured. Set WAQI_API_TOKEN as an environment variable.")
        return pd.DataFrame()

    url = f"https://api.waqi.info/map/bounds/?latlng={delhi_bounds}&token={token}"
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()
        if data.get("status") != "ok":
            return pd.DataFrame()
        df = pd.DataFrame(data.get("data", []))
        if df.empty:
            return df

        # Normalize columns
        if "aqi" in df.columns:
            df = df[df["aqi"] != "-"]
            df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
            df = df.dropna(subset=["aqi"])
        else:
            df["aqi"] = np.nan

        # station may be dict or string
        def _get_station_name(s):
            if isinstance(s, dict):
                return s.get("name") or s.get("station") or "N/A"
            return s or "N/A"

        def _get_time(s):
            if isinstance(s, dict):
                t = s.get("time", {})
                if isinstance(t, dict):
                    return t.get("s", "N/A")
                return t or "N/A"
            return "N/A"

        df["station_name"] = df.get("station", "").apply(_get_station_name)
        df["last_updated"] = df.get("station", "").apply(_get_time)
        df["lat"] = pd.to_numeric(df.get("lat", np.nan), errors="coerce")
        df["lon"] = pd.to_numeric(df.get("lon", np.nan), errors="coerce")
        df = df.dropna(subset=["lat", "lon"])

        # category/color/emoji/advice
        cats = df["aqi"].apply(lambda v: pd.Series(get_aqi_category(v), index=["category", "color", "emoji", "advice"]))
        df = pd.concat([df.reset_index(drop=True), cats.reset_index(drop=True)], axis=1)
        return df
    except Exception as e:
        st.error(f"Error fetching live AQI: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=1800)
def fetch_weather(lat=DELHI_LAT, lon=DELHI_LON):
    """Fetch current weather from Open-Meteo (free)."""
    try:
        url = f"{OPEN_METEO_BASE}/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&timezone=Asia%2FKolkata"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=3600)
def load_delhi_geojson(url=DELHI_GEOJSON_URL):
    """Load Delhi geojson if geopandas is available."""
    if gpd is None:
        return None, None
    try:
        gdf = gpd.read_file(url)
        gdf = gdf.to_crs(epsg=4326)
        polygon = gdf.unary_union
        return gdf, polygon
    except Exception:
        return None, None


# ------------------------
# Improved SMS logic & helpers
# ------------------------
if "LAST_SENT" not in st.session_state:
    st.session_state.LAST_SENT = {}  # phone -> timestamp

if "AQI_HISTORY" not in st.session_state:
    st.session_state.AQI_HISTORY = {}  # phone -> last weighted AQI


def get_nearby_stations_weighted(df, user_lat, user_lon, radius_km=10, max_stations=3):
    """Return nearest up to max_stations and weighted_aqi (inverse-distance)."""
    if df.empty:
        return pd.DataFrame(), None

    df = df.copy()
    df["distance_km"] = df.apply(lambda r: calculate_distance(user_lat, user_lon, r["lat"], r["lon"]), axis=1)
    nearby = df[df["distance_km"] <= radius_km].sort_values("distance_km").head(max_stations)

    if nearby.empty:
        return nearby, None

    # small epsilon to avoid division by zero
    nearby["weight"] = 1.0 / (nearby["distance_km"] + 0.01)
    weighted_aqi = (nearby["aqi"] * nearby["weight"]).sum() / nearby["weight"].sum()
    return nearby, float(weighted_aqi)


def detect_trend(phone, current_weighted_aqi, delta=TREND_DELTA):
    """Simple trend detection stored per phone (or user id)."""
    history = st.session_state.AQI_HISTORY.get(phone)
    st.session_state.AQI_HISTORY[phone] = current_weighted_aqi
    if history is None:
        return "Stable"
    if current_weighted_aqi > history + delta:
        return "Rising"
    if current_weighted_aqi < history - delta:
        return "Improving"
    return "Stable"


def can_send_sms(phone):
    """Rate limit per phone using session_state. Returns (bool, reason)"""
    now = time.time()
    last = st.session_state.LAST_SENT.get(phone)
    if last is None:
        return True, "OK"
    if now - last >= SMS_RATE_LIMIT_SECONDS:
        return True, "OK"
    remaining = int((SMS_RATE_LIMIT_SECONDS - (now - last)) / 60)
    return False, f"Rate limit: try again in ~{remaining} minutes"


def register_sms_sent(phone):
    st.session_state.LAST_SENT[phone] = time.time()


def format_weather_for_message(weather_json):
    if not weather_json:
        return "N/A"
    cw = weather_json.get("current_weather", {}) or weather_json.get("current", {})
    temp = cw.get("temperature") or cw.get("temperature_2m")
    code = cw.get("weathercode") or cw.get("weather_code") or None
    descr = "Weather"
    if code is not None:
        # simplified mapping
        try:
            code = int(code)
            if code == 0:
                descr = "Clear"
            elif code <= 3:
                descr = "Partly cloudy"
            elif code <= 55:
                descr = "Drizzle / Rain"
            elif code <= 65:
                descr = "Rain"
            elif code <= 82:
                descr = "Rain showers"
            else:
                descr = "Storm/Thunder"
        except Exception:
            descr = "Weather"
    if temp is None:
        return descr
    return f"{descr}, {temp:.1f}°C"


def build_alert_message(location_name, weighted_aqi, nearest_stations_df, weather_json, trend):
    cat, _, emoji, advice = get_aqi_category(weighted_aqi)
    parts = [
        f"🌍 Air Quality Alert - {location_name}",
        f"{emoji} AQI: {weighted_aqi:.0f} ({cat})",
        f"📊 Trend: {trend}",
        "",
        "Nearest stations:"
    ]
    for _, r in nearest_stations_df.iterrows():
        parts.append(f"- {r['station_name']}: {r['aqi']:.0f} ({r['distance_km']:.1f} km)")
    parts.append("")
    parts.append(f"🌦 {format_weather_for_message(weather_json)}")
    parts.append(f"💡 Advice: {advice}")
    return "\n".join(parts)


def validate_twilio_config():
    """Basic checks for Twilio config."""
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER):
        return False, "Twilio credentials not configured. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_PHONE_NUMBER in env."
    if not TWILIO_ACCOUNT_SID.startswith("AC"):
        return False, "Twilio Account SID seems invalid (should start with 'AC')."
    if not TWILIO_PHONE_NUMBER.startswith("+"):
        return False, "Twilio phone number must include country code and start with '+'."
    return True, "OK"


def send_sms_via_twilio(phone_number, message):
    """Send SMS using Twilio with robust error handling. Returns (success, message)."""
    try:
        from twilio.rest import Client
    except Exception:
        return False, "Twilio library is not installed. Run: pip install twilio"

    ok, reason = validate_twilio_config()
    if not ok:
        return False, reason

    # recipient validation (very basic)
    if not phone_number.startswith("+") or len(phone_number) < 8:
        return False, "Recipient phone number invalid. Include country code, e.g., +919876543210"

    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        sent = client.messages.create(body=message, from_=TWILIO_PHONE_NUMBER, to=phone_number)
        return True, f"Message sent (SID: {sent.sid})"
    except Exception as e:
        msg = str(e)
        if "authenticate" in msg.lower() or "401" in msg:
            return False, "Authentication failed with Twilio. Check Account SID & Auth Token."
        if "unverified" in msg.lower():
            return False, "Recipient number likely not verified for Twilio trial account."
        return False, f"Twilio Error: {msg}"


# ------------------------
# Visualization & UI functions
# ------------------------
def render_header(df):
    st.markdown('<div class="main-title">🌍 Delhi Air Quality Dashboard</div>', unsafe_allow_html=True)
    last = df["last_updated"].max() if (not df.empty and "last_updated" in df.columns) else "N/A"
    st.markdown(f'<div class="subtitle">Real-time monitoring • Last updated: {last}</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    if not df.empty:
        with c1:
            st.metric("Average AQI", f"{df['aqi'].mean():.1f}", delta=get_aqi_category(df['aqi'].mean())[0])
        with c2:
            min_idx = df["aqi"].idxmin()
            if pd.notna(min_idx):
                st.metric("Minimum AQI", f"{df['aqi'].min():.0f}", delta=f"{df.loc[min_idx, 'station_name']}")
            else:
                st.metric("Minimum AQI", "N/A")
        with c3:
            max_idx = df["aqi"].idxmax()
            if pd.notna(max_idx):
                st.metric("Maximum AQI", f"{df['aqi'].max():.0f}", delta=f"{df.loc[max_idx, 'station_name']}")
            else:
                st.metric("Maximum AQI", "N/A")
    else:
        with c1:
            st.write("No AQI data available.")

    with c4:
        weather = fetch_weather()
        if weather:
            cw = weather.get("current_weather") or weather.get("current")
            temp = cw.get("temperature") or cw.get("temperature_2m") if cw else None
            desc = format_weather_for_message(weather)
            st.markdown("<div style='background:#fff;padding:10px;border-radius:8px'>", unsafe_allow_html=True)
            st.subheader("Current Weather")
            st.write(desc)
            st.markdown("</div>", unsafe_allow_html=True)


def render_map_tab(df):
    st.subheader("📍 Live AQI Map")
    if df.empty:
        st.warning("No AQI station data available to render map.")
        return

    # pydeck scatter map
    st.pydeck_chart(
        pdk.Deck(
            map_style="carto-positron",
            initial_view_state=pdk.ViewState(latitude=DELHI_LAT, longitude=DELHI_LON, zoom=9.5, pitch=40),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df,
                    get_position='[lon, lat]',
                    get_fill_color='color',
                    get_radius=300,
                    pickable=True,
                    opacity=0.9,
                )
            ],
            tooltip={"html": "<b>{station_name}</b><br/>AQI: {aqi}<br/>Category: {category}", "style": {"color": "white"}},
        )
    )


def render_analytics_tab(df):
    st.subheader("📊 Analytics")
    if df.empty:
        st.info("No data to show analytics.")
        return
    c1, c2 = st.columns(2)
    with c1:
        st.write("AQI Category Distribution")
        fig = px.pie(values=df["category"].value_counts().values, names=df["category"].value_counts().index, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.write("Top 10 Polluted Stations")
        top10 = df.nlargest(10, "aqi").sort_values("aqi", ascending=True)
        fig2 = px.bar(top10, x="aqi", y="station_name", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)
    st.write("Full station data")
    st.dataframe(df[["station_name", "aqi", "category", "last_updated"]].sort_values("aqi", ascending=False))


def render_forecast_tab():
    st.subheader("📈 24-Hour AQI Forecast (Simulated)")
    hours = np.arange(0, 24)
    base = 120 + 40 * np.sin(hours / 3)
    noise = np.random.normal(0, 6, size=24)
    forecast = np.clip(base + noise, 40, 350)
    timestamps = [datetime.now() + timedelta(hours=int(h)) for h in hours]
    df = pd.DataFrame({"timestamp": timestamps, "aqi": forecast})
    fig = px.line(df, x="timestamp", y="aqi", markers=True)
    st.plotly_chart(fig, use_container_width=True)
    st.metric("Avg Forecast AQI (24h)", f"{df['aqi'].mean():.1f}")


def render_kriging_tab(df):
    st.subheader("🔥 Kriging Heatmap (masked to Delhi)")
    if perform_kriging_correct is None:
        st.warning("Kriging module not available. Skipping kriging heatmap.")
        return

    if df.empty:
        st.warning("No AQI data for kriging.")
        return

    if df["aqi"].nunique() < 2:
        st.error("Not enough AQI variability for kriging.")
        return

    if len(df) < 4:
        st.error("Need at least 4 stations for reliable kriging.")
        return

    gdf, polygon = load_delhi_geojson()
    bounds = (28.40, 28.88, 76.84, 77.35)
    with st.spinner("Running kriging..."):
        try:
            lon_grid, lat_grid, z = perform_kriging_correct(df, bounds)
            heatmap_df = pd.DataFrame({"lon": lon_grid.flatten(), "lat": lat_grid.flatten(), "aqi": z.flatten()})
            fig = px.density_mapbox(heatmap_df, lat="lat", lon="lon", z="aqi", radius=9, center={"lat": DELHI_LAT, "lon": DELHI_LON}, zoom=9, mapbox_style="carto-positron")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Kriging error: {e}")


# ------------------------
# Streamlit app body / layout
# ------------------------
def main():
    st.title("Delhi Air Quality Dashboard")
    df = fetch_live_aqi()
    render_header(df)

    if df.empty:
        st.error("Could not fetch live AQI data. Check WAQI API token or network.")
    tabs = st.tabs(["Live Map", "Alerts & Health", "Analytics", "SMS Alerts", "Forecast", "Kriging Heatmap"])
    with tabs[0]:
        render_map_tab(df)
    with tabs[1]:
        # Alerts & Health
        st.header("🔔 Alerts & Recommendations")
        if df.empty:
            st.warning("No data.")
        else:
            max_aqi = df["aqi"].max()
            cat, _, emoji, advice = get_aqi_category(max_aqi)
            st.info(f"Highest AQI: {max_aqi:.0f} — {cat} {emoji}\n\nAdvice: {advice}")
            # Show alert cards for Unhealthy+ categories
            for lvl, cond in [("Hazardous", df[df["aqi"] > 300]), ("Very Unhealthy", df[(df["aqi"] > 200) & (df["aqi"] <= 300)]), ("Unhealthy", df[(df["aqi"] > 150) & (df["aqi"] <= 200)])]:
                if not cond.empty:
                    st.markdown(f"**⚠️ {lvl} conditions detected**")
                    for _, r in cond.sort_values("aqi", ascending=False).iterrows():
                        st.markdown(f"- **{r['station_name']}** — AQI {r['aqi']:.0f}")

    with tabs[2]:
        render_analytics_tab(df)

    with tabs[3]:
        st.header("📱 SMS Alert Subscription (Advanced)")
        with st.form("sms_form"):
            col1, col2 = st.columns(2)
            with col1:
                location_name = st.text_input("Location name (for message)", "Your Area")
                user_lat = st.number_input("Latitude", value=28.6139, format="%.6f", step=0.0001, min_value=28.4, max_value=28.9)
                user_lon = st.number_input("Longitude", value=77.2090, format="%.6f", step=0.0001, min_value=76.8, max_value=77.4)
            with col2:
                phone_number = st.text_input("Phone number (include country code)", placeholder="+91XXXXXXXXXX")
                radius_km = st.slider("Search radius (km)", 1, 30, 10)
                send_now = st.form_submit_button("Send Alert Now")

        if send_now:
            if not phone_number or not phone_number.startswith("+"):
                st.error("Please enter a valid phone number including country code (e.g., +919876543210).")
            else:
                # find nearby stations (weighted)
                nearby, weighted = get_nearby_stations_weighted(df, user_lat, user_lon, radius_km=radius_km, max_stations=3)
                if nearby.empty or weighted is None:
                    st.warning("No nearby AQI stations found within radius. Try increasing radius.")
                else:
                    # trend detection & rate limiting
                    trend = detect_trend(phone_number, weighted)
                    can_send, reason = can_send_sms(phone_number)
                    if not can_send:
                        st.warning(reason)
                    else:
                        # threshold based decision: send if weighted crosses healthy thresholds or if severity high
                        prev = st.session_state.AQI_HISTORY.get(phone_number, 0)
                        thresholds = [100, 150, 200, 300]
                        should_send = False
                        # send if crossed any threshold from previous to current OR current above 150 (high severity)
                        for t in thresholds:
                            if prev < t <= weighted:
                                should_send = True
                                break
                        if not should_send and weighted >= 150 and (prev == 0 or weighted >= prev):
                            # High severity - allow send
                            should_send = True

                        if not should_send:
                            st.info("No significant threshold change detected and severity not high — not sending SMS to avoid spam.")
                        else:
                            weather = fetch_weather(user_lat, user_lon)  # weather near user
                            message = build_alert_message(location_name or "Your Area", weighted, nearby, weather, trend)
                            success, resp = send_sms_via_twilio(phone_number, message)
                            if success:
                                register_sms_sent(phone_number)
                                st.success(f"SMS sent successfully: {resp}")
                                st.text_area("Message Sent (preview)", message, height=220)
                            else:
                                st.error(f"Failed to send SMS: {resp}")
                                st.text_area("Message (not sent)", message, height=220)

        st.markdown("---")
        st.write("Notes:")
        st.write("- To enable Twilio SMS: set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER in environment variables.")
        st.write("- Twilio trial accounts require verifying recipient phone numbers in Twilio console.")

    with tabs[4]:
        render_forecast_tab()

    with tabs[5]:
        render_kriging_tab(df)


if __name__ == "__main__":
    main()
