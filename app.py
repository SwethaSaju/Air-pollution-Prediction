import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import pydeck as pdk
from datetime import datetime, timedelta
from shapely.geometry import Polygon, Point
import plotly.express as px

############################################
#          ENVIRONMENT VARIABLES
############################################
FAST2SMS_API_KEY = os.environ.get("FAST2SMS_API_KEY", "")
WAQI_API_TOKEN = os.environ.get("WAQI_API_TOKEN", "")

############################################
#          HELPER FUNCTIONS
############################################

def fetch_live_aqi_data():
    """Fetch live AQI data from WAQI"""
    if not WAQI_API_TOKEN:
        return None, "Missing WAQI API token"

    url = f"https://api.waqi.info/map/bounds/?latlng=28.404,76.840,28.883,77.349&token={WAQI_API_TOKEN}"
    
    try:
        r = requests.get(url, timeout=10)
        j = r.json()
        
        if j.get("status") != "ok":
            return None, "WAQI API Error: " + str(j)

        data = j["data"]
        df = pd.DataFrame(data)
        df = df.rename(columns={"lat": "lat", "lon": "lon", "aqi": "aqi", "uid": "uid"})
        df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")
        df = df.dropna(subset=["aqi"])

        # Fetch station names
        names = []
        for uid in df["uid"]:
            try:
                s_url = f"https://api.waqi.info/feed/@{uid}/?token={WAQI_API_TOKEN}"
                s = requests.get(s_url).json()
                name = s["data"]["city"]["name"]
            except:
                name = "Unknown Station"
            names.append(name)

        df["station_name"] = names
        return df, None

    except Exception as e:
        return None, "Error fetching AQI data: " + str(e)



def fetch_weather():
    """Fetch Delhi Live Weather Using Open-Meteo"""
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=28.6&longitude=77.2&current_weather=true"
        r = requests.get(url, timeout=10).json()
        temp = r["current_weather"]["temperature"]
        wind = r["current_weather"]["windspeed"]
        return f"{temp}°C, Wind {wind} km/h"
    except:
        return "Weather unavailable"


def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine distance in KM"""
    R = 6371
    p = np.radians
    d = p(lat2 - lat1), p(lon2 - lon1)
    a = np.sin(d[0]/2)**2 + np.cos(p(lat1))*np.cos(p(lat2))*np.sin(d[1]/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


############################################
#       FAST2SMS — SMS SENDER
############################################

def send_sms_fast2sms(phone_number, message):
    """
    Sends SMS using Fast2SMS API (Indian numbers only)
    """
    if not FAST2SMS_API_KEY:
        return False, "FAST2SMS API key missing"

    # Normalize phone number: +91XXXXXXXXXX → XXXXXXXXXX
    phone_number = phone_number.replace("+91", "").strip()

    if not (phone_number.isdigit() and len(phone_number) == 10):
        return False, "Enter valid Indian mobile number (10 digits)."

    url = "https://www.fast2sms.com/dev/bulkV2"

    payload = {
        "message": message,
        "language": "english",
        "route": "v3",
        "numbers": phone_number,
    }

    headers = {
        "authorization": FAST2SMS_API_KEY
    }

    try:
        resp = requests.get(url, params=payload, headers=headers)
        data = resp.json()

        if data.get("return") == True:
            return True, "SMS sent successfully!"
        else:
            return False, str(data)

    except Exception as e:
        return False, f"Error sending SMS: {e}"


############################################
#       AQI CATEGORY FUNCTION
############################################

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "🟢", "Air quality is ideal."
    elif aqi <= 100:
        return "Moderate", "🟡", "Acceptable air quality."
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "Sensitive groups should reduce activity."
    elif aqi <= 200:
        return "Unhealthy", "🔴", "General public should reduce outdoor activity."
    elif aqi <= 300:
        return "Very Unhealthy", "🟣", "Everyone may experience health effects."
    else:
        return "Hazardous", "⚫", "Avoid going outdoors."


############################################
#                 UI
############################################

st.set_page_config(page_title="Delhi AQI Monitor", layout="wide")
st.title("🌫️ Delhi AI-Driven AQI Monitoring with SMS Alerts")

st.write("Live AQI, Weather, Hotspots & Fast2SMS Alerts")

############################################
# Fetch AQI Data
############################################

with st.spinner("Fetching live AQI data…"):
    df, err = fetch_live_aqi_data()

if err:
    st.error(err)
    st.stop()

st.success("Live AQI data loaded!")

############################################
# Show Map
############################################

st.subheader("🗺 Live AQI Map of Delhi")

df["color"] = df["aqi"].apply(
    lambda x: [255, 0, 0] if x > 200 else ([255, 165, 0] if x > 150 else ([0, 200, 0]))
)

layer = pdk.Layer(
    "ScatterplotLayer",
    df,
    get_position='[lon, lat]',
    get_radius=300,
    get_fill_color="color",
)

view_state = pdk.ViewState(latitude=28.63, longitude=77.22, zoom=10)

st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state))

############################################
# Analytics
############################################
st.subheader("📊 AQI Analytics")

fig = px.histogram(df, x="aqi", nbins=20, title="AQI Distribution")
st.plotly_chart(fig, use_container_width=True)

############################################
# SMS ALERT SYSTEM
############################################

st.subheader("📱 Send AQI Alert via Fast2SMS")

user_location = st.text_input("Enter your location (e.g., Connaught Place):")
user_phone = st.text_input("Enter your mobile number (+91XXXXXXXXXX):")

if st.button("Send SMS Alert"):
    if not user_location or not user_phone:
        st.warning("Please enter both location and phone number.")
    else:
        # Get nearest station
        df["distance"] = df.apply(
            lambda row: calculate_distance(28.63, 77.22, row["lat"], row["lon"]), axis=1
        )

        nearest = df.sort_values("distance").iloc[0]

        category, emoji, advice = get_aqi_category(nearest["aqi"])

        message = (
            f"AQI Alert - {user_location}\n"
            f"{emoji} AQI: {nearest['aqi']} ({category})\n"
            f"Nearest Station: {nearest['station_name']}\n"
            f"Health Advice: {advice}"
        )

        success, status_message = send_sms_fast2sms(user_phone, message)

        if success:
            st.success(status_message)
            st.info(f"Message Sent:\n{message}")
        else:
            st.error(status_message)




############################################
# Data Table
############################################

st.subheader("📋 Live AQI Data Table")
st.dataframe(df)

