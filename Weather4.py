import requests
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from prophet import Prophet
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import base64
import json

# ----------------------------
# Auto-refresh every 30 mins
# ----------------------------
st_autorefresh(interval=30*60*1000, key="auto_refresh")  # 30 minutes

# ----------------------------
# GitHub config for Community Reports
# ----------------------------
GITHUB_USER = st.secrets["GITHUB_USER"]
GITHUB_REPO = st.secrets["GITHUB_REPO"]
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
JSON_PATH = "reports.json"  # path in repo
IMAGES_FOLDER = "images"

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

# ----------------------------
# Helper functions for GitHub
# ----------------------------
def get_reports():
    """Fetch reports.json from GitHub"""
    url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{JSON_PATH}"
    r = requests.get(url, headers=HEADERS)
    if r.status_code == 200:
        content = r.json()
        file_sha = content.get("sha")
        data = base64.b64decode(content.get("content", "")).decode()
        if not data.strip():
            return [], file_sha  # empty file, return empty list
        try:
            return json.loads(data), file_sha
        except json.JSONDecodeError:
            return [], file_sha  # corrupted content, return empty list
    else:
        return [], None  # file does not exist yet

def update_reports(new_report):
    # Upload image first
    file_name = f"{IMAGES_FOLDER}/{int(datetime.now().timestamp())}_{new_report['image_file'].name}"
    file_bytes = new_report['image_file'].read()
    encoded_image = base64.b64encode(file_bytes).decode()
    url_upload = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{file_name}"
    r = requests.put(url_upload, headers=HEADERS, json={
        "message": f"Upload image {file_name}",
        "content": encoded_image
    })
    if r.status_code not in [200, 201]:
        st.error("Failed to upload image to GitHub")
        st.stop()
    
    image_url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/main/{file_name}"
    new_report['image_url'] = image_url
    del new_report['image_file']

    # Update JSON
    reports, sha = get_reports()
    new_report["id"] = max([r.get("id", 0) for r in reports]+[0]) + 1
    new_report["comments"] = []
    new_report["timestamp"] = datetime.now().isoformat()
    reports.append(new_report)

    encoded_json = base64.b64encode(json.dumps(reports, indent=2).encode()).decode()
    url_json = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{JSON_PATH}"
    r2 = requests.put(url_json, headers=HEADERS, json={
        "message": "Update reports.json",
        "content": encoded_json,
        "sha": sha
    })
    if r2.status_code not in [200, 201]:
        st.error("Failed to update reports.json")
        st.stop()

# ----------------------------
# Weather & Flood Predictor functions
# ----------------------------
def fetch_weather(latitude, longitude):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m,rain&timezone=Asia/Kolkata"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return None

def fetch_historical_weather(latitude, longitude, days_back=14):
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)
    delta = timedelta(days=7)
    all_times, all_rain = [], []
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + delta, end_date)
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}"
               f"&hourly=temperature_2m,rain&start={current_start.strftime('%Y-%m-%dT%H:%M')}"
               f"&end={current_end.strftime('%Y-%m-%dT%H:%M')}&timezone=Asia/Kolkata")
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            all_times.extend(data['hourly']['time'])
            all_rain.extend(data['hourly']['rain'])
        except:
            pass
        current_start = current_end + timedelta(days=1)
    return pd.DataFrame({"ds": pd.to_datetime(all_times), "y": all_rain})

def prepare_data_for_prophet(lat, lon):
    df_hist = fetch_historical_weather(lat, lon, 14)
    fut_data = fetch_weather(lat, lon)
    if not fut_data or df_hist is None:
        return None, None
    fut_times = fut_data['hourly']['time']
    fut_rain = fut_data['hourly']['rain']
    df_combined = pd.concat([df_hist, pd.DataFrame({"ds": pd.to_datetime(fut_times), "y": fut_rain})])
    return df_combined, fut_times[:3]

def prophet_forecast_with_history(df, hours_ahead=3):
    if len(df) < 24:
        return df['y'].tail(hours_ahead).values
    model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=hours_ahead, freq='H')
    forecast = model.predict(future)
    return forecast[['yhat']].tail(hours_ahead)['yhat'].clip(lower=0).values

def predict_flood_risk_from_rain(rain_values):
    total_rain = sum(rain_values)
    if total_rain > 20:
        return "High", total_rain, min(100, total_rain*3)
    elif total_rain > 10:
        return "Medium", total_rain, min(80, total_rain*2.5)
    else:
        return "Low", total_rain, min(50, total_rain*2)

def rain_probability(predicted_rain):
    total_rain = sum(predicted_rain)
    if total_rain <= 0.1: return 0
    elif total_rain < 1: return 30
    elif total_rain < 5: return 60
    else: return 90

# ----------------------------
# Streamlit Setup
# ----------------------------
st.set_page_config(page_title="Weather & Flood Predictor + Community", layout="wide")
st.title("🌦️ Weather & Flood Predictor (14-day history) & Community Reports")

cities_coords = {
    "Mumbai": (19.0760, 72.8777),
    "Delhi": (28.6139, 77.2090),
    "Bangalore": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Chhattisgarh": (21.2951,81.8282)
}

dashboard_type = st.sidebar.radio("Select Dashboard", ["Citizen/User", "Admin", "Community Reports"])

# ----------------------------
# Citizen/User
# ----------------------------
if dashboard_type == "Citizen/User":
    st.header("🌍 Citizen Weather Alerts")
    city = st.selectbox("Select your city", list(cities_coords.keys()))
    lat, lon = cities_coords[city]

    df, next_3_times = prepare_data_for_prophet(lat, lon)
    if df is not None:
        predicted_rain = prophet_forecast_with_history(df, 3)
        rain_prob = rain_probability(predicted_rain)
        risk, _, flood_prob = predict_flood_risk_from_rain(predicted_rain)

        st.subheader(f"Flood Risk in {city}: {risk}")
        st.write(f"☔ Chance of Rain in next 3 hours: {rain_prob}%")
        st.write(f"💧 Approximate Flood Probability: {flood_prob:.0f}%")
        st.write("🕒 Prediction times (local):", ", ".join(next_3_times))

        m = folium.Map(location=[lat, lon], zoom_start=10)
        folium.CircleMarker(
            location=[lat, lon],
            radius=20,
            color="red" if risk=="High" else "orange" if risk=="Medium" else "green",
            fill=True,
            fill_color="red" if risk=="High" else "orange" if risk=="Medium" else "green",
            popup=f"{city}\nRisk: {risk}\nRain Chance: {rain_prob}%\nFlood Prob: {flood_prob:.0f}%"
        ).add_to(m)
        st_folium(m, width=700, height=500)
    else:
        st.error("Failed to fetch weather data")

# ----------------------------
# Admin
# ----------------------------
elif dashboard_type == "Admin":
    st.header("🛠️ Admin Risk Reports")
    risks = []
    for city_name, (lat, lon) in cities_coords.items():
        df, next_3_times = prepare_data_for_prophet(lat, lon)
        if df is not None:
            predicted_rain = prophet_forecast_with_history(df, 3)
            rain_prob = rain_probability(predicted_rain)
            risk, _, flood_prob = predict_flood_risk_from_rain(predicted_rain)
        else:
            risk, rain_prob, flood_prob = "Unknown", 0, 0
        risks.append({
            "City": city_name,
            "Risk": risk,
            "Rain_%": rain_prob,
            "Flood_%": flood_prob,
            "Lat": lat,
            "Lon": lon,
            "Next_3_hours": ", ".join(next_3_times) if next_3_times else "N/A"
        })
    df_risks = pd.DataFrame(risks)
    st.dataframe(df_risks[['City','Risk','Rain_%','Flood_%','Next_3_hours']])

    m = folium.Map(location=[20, 78], zoom_start=5)
    for entry in risks:
        folium.CircleMarker(
            location=[entry["Lat"], entry["Lon"]],
            radius=15,
            color="red" if entry["Risk"]=="High" else "orange" if entry["Risk"]=="Medium" else "green",
            fill=True,
            fill_color="red" if entry["Risk"]=="High" else "orange" if entry["Risk"]=="Medium" else "green",
            popup=f"{entry['City']}\nRisk: {entry['Risk']}\nRain Chance: {entry['Rain_%']}%\nFlood Prob: {entry['Flood_%']}%"
        ).add_to(m)
    st_folium(m, width=700, height=500)

# ----------------------------
# Community Reports
# ----------------------------
else:
   st.header("📸 Community Weather Reports")

# Ask for user name once
   current_user = st.text_input("Enter your name (for posting/deleting)", key="current_user")

# ----------------------------
# Post a new report
# ----------------------------
   with st.form("report_form"):
       caption = st.text_area("Add a caption about current weather")
       uploaded_img = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
       submitted = st.form_submit_button("Post Report")
       if submitted:
           if not current_user or not caption or not uploaded_img:
               st.error("Please fill all fields and upload an image")
           else:
               new_report = {
                   "name": current_user.strip(),
                   "caption": caption.strip(),
                   "image_file": uploaded_img
                }
               update_reports(new_report)
               st.success("✅ Report posted successfully!")

   st.markdown("---")
   st.subheader("All Community Reports")
   reports, sha = get_reports()

   if reports:
       for report in sorted(reports, key=lambda x: x["timestamp"], reverse=True):
            st.image(report["image_url"], use_container_width=True)
            st.write(f"**{report['name']}**: {report['caption']}")
            st.write(f"_Posted at {report['timestamp']}_")

        # ----------------------------
        # Delete post button (only for the uploader)
        # ----------------------------
            if current_user.strip() == report["name"]:
               if st.button(f"Delete Post", key=f"delpost_{report['id']}"):
                   reports = [r for r in reports if r["id"] != report["id"]]
                   # Update JSON in GitHub
                   encoded_json = base64.b64encode(json.dumps(reports, indent=2).encode()).decode()
                   url_json = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{JSON_PATH}"
                   r2 = requests.put(url_json, headers=HEADERS, json={
                       "message": f"Delete report {report['id']}",
                       "content": encoded_json,
                       "sha": sha
                   })
                   if r2.status_code in [200, 201]:
                       st.success("✅ Post deleted successfully")
                       st.experimental_rerun()
                   else:
                       st.error("Failed to delete post")

        # ----------------------------
        # Show comments
        # ----------------------------
            if report.get("comments"):
                st.write("💬 Comments:")
                for idx, c in enumerate(report["comments"]):
                    st.write(f"- {c}")
                # Delete comment button (only if current user owns it)
                    comment_owner = c.split(":")[0].strip()
                    if current_user.strip() == comment_owner:
                        if st.button(f"Delete Comment", key=f"delcomment_{report['id']}_{idx}"):
                            report["comments"].pop(idx)
                        # Update JSON in GitHub
                            encoded_json = base64.b64encode(json.dumps(reports, indent=2).encode()).decode()
                            r2 = requests.put(url_json, headers=HEADERS, json={
                                "message": f"Delete comment {idx} on report {report['id']}",
                                "content": encoded_json,
                                "sha": sha
                            })
                            if r2.status_code in [200, 201]:
                                st.success("✅ Comment deleted")
                                st.experimental_rerun()
                            else:
                                st.error("Failed to delete comment")

        # ----------------------------
        # Add comment form
        # ----------------------------
            with st.form(f"comment_form_{report['id']}"):
                comment_text = st.text_input("Add a comment", key=f"comment_text_{report['id']}")
                comment_submitted = st.form_submit_button("Post Comment")
                if comment_submitted:
                    if not current_user or not comment_text:
                        st.error("Enter both name and comment")
                    else:
                        report.setdefault("comments", []).append(f"{current_user.strip()}: {comment_text.strip()}")
                        # Update JSON in GitHub
                        encoded_json = base64.b64encode(json.dumps(reports, indent=2).encode()).decode()
                        r2 = requests.put(url_json, headers=HEADERS, json={
                            "message": f"Add comment on report {report['id']}",
                            "content": encoded_json,
                            "sha": sha
                            })
                        if r2.status_code in [200, 201]:
                            st.success("✅ Comment posted")
                            st.experimental_rerun()
                        else:
                            st.error("Failed to post comment")
