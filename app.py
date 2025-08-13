import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import os
import subprocess
import platform
from geopy.geocoders import Nominatim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- PAGE CONFIG ---
st.set_page_config(page_title="Signal Strength Mapping", layout="wide")

# --- SIDEBAR: CSV Upload ---
st.sidebar.header("📂 Data Settings")
uploaded_csv = st.sidebar.file_uploader("Upload CSV", type=["csv"])
default_csv_path = "training.csv"

if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
else:
    df = pd.read_csv(default_csv_path)

# --- HELPER FUNCTIONS ---
def classify_signal(value):
    if value == 0:
        return "No Signal"
    elif value <= 50:
        return "Poor"
    elif value <= 80:
        return "Medium"
    else:
        return "Excellent"

df['Signal_Category'] = df['Signal'].apply(classify_signal)

def reverse_geocode(lat, lon):
    geolocator = Nominatim(user_agent="signal_app")
    location = geolocator.reverse((lat, lon), language="en")
    return location.address if location else "Location not found"

def geocode_address(address):
    geolocator = Nominatim(user_agent="signal_strength_app")
    try:
        location = geolocator.geocode(address, timeout=10)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None

def get_windows_wifi_strength():
    if platform.system() == "Windows":
        try:
            result = subprocess.check_output(
                ['netsh', 'wlan', 'show', 'interfaces'],
                shell=True
            ).decode('utf-8', errors="ignore")
            for line in result.split("\n"):
                if "Signal" in line:
                    return line.split(":")[1].strip()
        except:
            return "Unable to detect Wi-Fi strength"
    return "Wi-Fi strength detection works only on Windows"

# --- MAP GENERATOR FUNCTION ---
def generate_coverage_map(csv_path, output_html):
    df_map = pd.read_csv(csv_path)

    def color_for_signal(signal):
        if signal == 0:
            return "black"
        elif signal == 100:
            return "green"
        elif signal == 80:
            return "yellow"
        elif signal == 50:
            return "red"
        else:
            return "gray"

    m = folium.Map(location=[df_map["Latitude"].mean(), df_map["Longitude"].mean()], zoom_start=5)

    categories = {
        "Green (Excellent)": df_map[df_map["Signal"] == 100],
        "Yellow (Good)": df_map[df_map["Signal"] == 80],
        "Red (Poor)": df_map[df_map["Signal"] == 50],
        "Black (No Signal)": df_map[df_map["Signal"] == 0]
    }

    for name, data in categories.items():
        cluster = MarkerCluster(name=name, disableClusteringAtZoom=8).add_to(m)
        for _, row in data.iterrows():
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=5,
                color=color_for_signal(row["Signal"]),
                fill=True,
                fill_color=color_for_signal(row["Signal"]),
                fill_opacity=0.6,
                popup=f"""
                    <b>Cell ID:</b> {row['Cell_ID']}<br>
                    <b>Network:</b> {row['Network_Type']}<br>
                    <b>Signal:</b> {row['Signal']} dBm
                """
            ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_html)


# --- MAIN APP: TABS ---
tabs = st.tabs([
    " Project Info ",
    " Signal Strength Analysis",
    " Coverage Map",
    " Coverage Gap Detection",
    " Geocoding Tools",
    " Windows Wi-Fi Strength"
])

# --- TAB 1: Project Info ---
with tabs[0]:
    st.title(" Project: Signal Strength Mapping & Coverage Gap Detection")

    st.markdown("""
    **Description**  
    The **Signal Strength & Coverage Analysis Tool** is a Windows-based interactive application designed to help users **visualize, analyze, and interpret mobile network performance** in a clear and intuitive way.  

    """)

    st.subheader("✨ Key Features")
    st.markdown("""
    -  **Flexible Data Input** – Load the default DATA.csv or upload your own dataset from the sidebar  
    -  **Signal Strength Analysis** – Explore average, maximum, and minimum signal levels, with visual charts  
    -  **Coverage Map Visualization** – View an interactive map (map.html) with a slightly zoomed-out view  
    -  **Coverage Gap Detection** – Highlight problem zones where signal falls below thresholds  
    -  **Reverse Geocoding** – Get the address for given latitude & longitude instantly  
    -  **Wi-Fi Signal Strength (Windows only)** – Detect and display your PC’s current Wi-Fi signal quality  
    """)

    st.subheader(" Usage Guide")
    st.markdown("""
    1. **Upload a CSV** from the sidebar or use the default DATA.csv  
    2. Navigate between tabs to explore analysis options  
    3. **Signal Categories**:  
       -  **0**: No Signal   (black) 
       -  **50**: Poor       (red)
       -  **80**: Medium     (yellow)
       -  **100**: Excellent (green) 
    """)

    st.subheader(" Required Columns in CSV")
    st.markdown("""
    | Column Name                 | Description                                  | Example Value            |
    |-----------------------------|----------------------------------------------|--------------------------|
    | **Network_Type**            | Mobile network technology                    | 4G,5G                    |
    | **MCC**                     | Mobile Country Code (Country Identifier)     | 404 (India)              |
    | **MNC**                     | Mobile Network Code (Operator Identifier)    | 10 (Airtel)              |
    | **LAC**                     | Location Area Code                           | 23456                    |
    | **Cell_ID**                 | Unique identifier for the cell tower         | 12345678                 |
    | **Signal**                  | Signal strength in dBm                       | -85                      |
    | **Longitude**               | Longitude coordinate                         | 77.5946                  |
    | **Latitude**                | Latitude coordinate                          | 12.9716                  |
    | **Timestamp1**              | Measurement date/time                        | 2025-08-12 14:30:00      |
    | **Timestamp2** *(optional)* | End time of measurement                      | 2025-08-12 14:35:00      |
    """)

# --- TAB 2: Signal Strength Analysis ---
with tabs[1]:
    st.header("📊 Signal Strength Analysis")
    st.subheader("Raw Data")
    st.dataframe(df)

    st.subheader("Signal Strength Distribution")
    st.bar_chart(df['Signal'].value_counts())

    st.subheader("Model Test on CSV")
    target_col = "Signal"
    label_encoders = {}
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            label_encoders[col] = le

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"### ✅ Model Accuracy: {acc*100:.2f}%")
    st.text("Classification Report:\n" + classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# --- TAB 3: Coverage Map ---
with tabs[2]:
    st.header("Coverage Map")
    map_path = "map.html"

    df.to_csv("temp_for_map.csv", index=False)
    generate_coverage_map("temp_for_map.csv", map_path)


    if os.path.exists(map_path):
        st.components.v1.html(open(map_path, 'r').read(), height=600, scrolling=False)
    else:
        st.error("Map file not found.")

# --- TAB 4: Coverage Gap Detection ---
with tabs[3]:
    st.header("Coverage Gap Detection")

    def generate_coverage_gap_map(df):
        # Filter for Poor/No Signal
        gap_df = df[df['Signal'] <= 50]
        if gap_df.empty:
            return None

        # Create base map
        m = folium.Map(location=[gap_df["Latitude"].mean(), gap_df["Longitude"].mean()], zoom_start=5)

        # Create a cluster group for coverage gaps
        gap_cluster = MarkerCluster(name="Coverage Gaps", disableClusteringAtZoom=8).add_to(m)

        for _, row in gap_df.iterrows():
            color = "black" if row["Signal"] == 0 else "red"
            folium.CircleMarker(
                location=[row["Latitude"], row["Longitude"]],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=f"""
                    <b>Cell ID:</b> {row['Cell_ID']}<br>
                    <b>Network:</b> {row['Network_Type']}<br>
                    <b>Signal:</b> {row['Signal']} dBm
                """
            ).add_to(gap_cluster)

        folium.LayerControl(collapsed=False).add_to(m)

        map_path = "coverage_gap_map.html"
        m.save(map_path)
        return map_path

    # Generate and display the gap map
    map_path = generate_coverage_gap_map(df)
    if map_path:
        st.components.v1.html(open(map_path, 'r', encoding="utf-8").read(), height=500)
    else:
        st.warning("No coverage gaps found in the current dataset.")


# --- TAB 5: Geocoding Tools ---
with tabs[4]:
    st.header("📍 Reverse Geocoding (Lat/Lon → Address)")
    lat_input = st.number_input("Latitude", format="%.6f", key="rev_lat")
    lon_input = st.number_input("Longitude", format="%.6f", key="rev_lon")
    if st.button("Get Address"):
        st.success(reverse_geocode(lat_input, lon_input))

    st.markdown("---")
    st.header("📍 Forward Geocoding (Address → Lat/Lon)")
    address_input = st.text_input("Enter location address:", "")
    if st.button("Get Coordinates"):
        if address_input.strip():
            geo_lat, geo_lon = geocode_address(address_input)
            if geo_lat and geo_lon:
                st.success(f"Latitude: {geo_lat}, Longitude: {geo_lon}")
                st.map(pd.DataFrame({"lat": [geo_lat], "lon": [geo_lon]}))
            else:
                st.error("Address not found.")
        else:
            st.warning("Please enter an address.")

# --- TAB 6: Wi-Fi Strength (Windows) ---
with tabs[5]:
    st.header("Windows Wi-Fi Strength")
    st.info("Only works on Windows PC.")
    if st.button("Check Wi-Fi Strength"):
        st.success(get_windows_wifi_strength())
