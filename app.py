import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import os
import subprocess
import platform
from geopy.geocoders import Nominatim

# ML & plotting
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Signal Strength Mapping", layout="wide")

# --- SIDEBAR: CSV Upload ---
st.sidebar.header("📂 Data Settings")
uploaded_csv = st.sidebar.file_uploader("Upload CSV", type=["csv"], key="sidebar_csv")
default_csv_path = "training.csv"

if uploaded_csv is not None:
    df = pd.read_csv(uploaded_csv)
else:
    df = pd.read_csv(default_csv_path)

# --- HELPER FUNCTIONS ---
def classify_signal(value):
    # Assumes 'Signal' scaled like 0, 50, 80, 100 (your dataset style)
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
        except Exception:
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
                    <b>Signal:</b> {row['Signal']}
                """
            ).add_to(cluster)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(output_html)

# --- ML: Train regression model from current df for prediction tab ---
def train_regression_model(current_df):
    """
    Trains a RandomForestRegressor to predict numeric Signal.
    Uses available features among: Latitude, Longitude, MCC, MNC, LAC, Cell_ID, Network_Type.
    Returns (model, feature_list, label_encoder_for_network, defaults).
    """
    df_train = current_df.copy().dropna(subset=["Signal", "Latitude", "Longitude"])
    feature_candidates = ["Latitude", "Longitude", "MCC", "MNC", "LAC", "Cell_ID", "Network_Type"]
    features = [c for c in feature_candidates if c in df_train.columns]

    if "Network_Type" in features:
        # Encode Network_Type
        le = LabelEncoder()
        df_train["Network_Type"] = df_train["Network_Type"].astype(str)
        df_train["Network_Type_enc"] = le.fit_transform(df_train["Network_Type"])
        features = [f for f in features if f != "Network_Type"] + ["Network_Type_enc"]
    else:
        le = None

    X = df_train[features].copy()
    y = df_train["Signal"].astype(float)

    # Fill NaNs
    for col in X.columns:
        if X[col].isna().any():
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode().iat[0])

    # Train model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)

    # Save defaults for later use in prediction
    feature_defaults = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            feature_defaults[col] = X[col].median()
        else:
            feature_defaults[col] = X[col].mode().iat[0]

    return model, features, le, feature_defaults


def predict_signal_strength_with_artifacts(lat, lon, artifacts):
    """
    Predicts signal strength using only Latitude & Longitude from user,
    other features auto-filled with defaults learned during training.
    """
    model, features, le, defaults = (
        artifacts["model"], artifacts["features"], artifacts["le"], artifacts["defaults"]
    )

    # Start with lat/lon
    row = {"Latitude": float(lat), "Longitude": float(lon)}

    # Fill missing features with defaults
    for col in features:
        if col not in row:
            row[col] = defaults[col]

    input_df = pd.DataFrame([row])

    # Handle Network_Type encoding if present
    if "Network_Type_enc" in features:
        input_df["Network_Type_enc"] = defaults["Network_Type_enc"]

    # Subset to features used in training
    X_pred = input_df[features].copy()

    # Ensure numeric
    for c in X_pred.columns:
        X_pred[c] = pd.to_numeric(X_pred[c], errors="coerce").fillna(0)

    pred_value = float(model.predict(X_pred)[0])

    # Map to category
    if pred_value == 0:
        category, color = "No Signal", "black"
    elif pred_value <= 70:
        category, color = "Poor", "red"
    elif pred_value <= 88:
        category, color = "Medium", "yellow"
    else:
        category, color = "Excellent", "green"

    return pred_value, category, color


# --- Store artifacts in session (only once after training) ---
if "pred_artifacts" not in st.session_state or st.session_state["pred_artifacts"] is None:
    model, feat_list, net_le, defaults = train_regression_model(df)
    st.session_state["pred_artifacts"] = {
        "model": model,
        "features": feat_list,
        "le": net_le,
        "defaults": defaults,
    }
def plot_signal_distribution(df):
    # Map categories
    def categorize_signal(val):
        if val == 0:
            return "No Signal"
        elif val <= 50:
            return "Poor"
        elif val <= 80:
            return "Medium"
        else:
            return "Excellent"

    df["Signal_Category"] = df["Signal"].apply(categorize_signal)

    # Count values
    counts = df["Signal_Category"].value_counts().reindex(
        ["No Signal", "Poor", "Medium", "Excellent"], fill_value=0
    )

    # Better color palette
    category_colors = {
        "No Signal": "#4B4B4B",   # dark gray
        "Poor": "#E74C3C",        # red
        "Medium": "#F1C40F",      # gold
        "Excellent": "#2ECC71"    # green
    }

    # Plot bar chart
    fig, ax = plt.subplots(figsize=(5,4))
    bars = sns.barplot(
        x=counts.index,
        y=counts.values,
        palette=[category_colors[c] for c in counts.index],
        ax=ax
    )

    ax.set_title("Signal Strength Distribution", fontsize=8, fontweight="bold")
    ax.set_ylabel("Count", fontsize=5)
    ax.set_xlabel("Signal Category", fontsize=5)

    # Add value labels on bars
    for i, bar in enumerate(bars.patches):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold"
        )

    st.pyplot(fig)


# --- MAIN APP: TABS ---
tabs = st.tabs([
    " Project Info ",
    " Signal Strength Analysis",
    " Coverage Map",
    " Coverage Gap Detection",
    " Recommendation ",
    " Geocoding Tools",  
    " Windows Wi-Fi Strength"
])

# --- TAB 1: Project Info ---
with tabs[0]:
    st.title(" Data Driven : Signal Strength Mapping & Coverage Gap Detection ")

    st.markdown("""
**Description**  
The **Signal Strength & Coverage Analysis Tool** is a **data-driven** interactive application designed to **visualize, analyze, and interpret** mobile network performance.  
                Note: It is research-based and uses synthetic data
""")
    st.subheader(" Usage Guide")
    st.markdown("""
1. **Upload a CSV** from the sidebar or use the default CSV  
        Note : The CSV file should contain the following header
2. Navigate between tabs to explore analysis options  
3. **Synthetic Signal Categories**:  
   - **0**: No Signal (black)  
   - **50**: Poor (red)  
   - **80**: Medium (yellow)  
   - **100**: Excellent (green)  
""")

    st.subheader(" Required header in CSV")
    st.markdown("""
| Header Name                 | Description                                  | Example Value            |
|-----------------------------|----------------------------------------------|--------------------------|
| **Network_Type**            | Mobile network technology                    | 4G, 5G                   |
| **MCC**                     | Mobile Country Code (Country Identifier)     | 404 (India)              |
| **MNC**                     | Mobile Network Code (Operator Identifier)    | 10 (Airtel)              |
| **LAC**                     | Location Area Code                           | 23456                    |
| **Cell_ID**                 | Unique identifier for the cell tower         | 12345678                 |
| **Signal**                  | Signal strength (e.g., 0/50/80/100)          | 80                       |
| **Longitude**               | Longitude coordinate                         | 77.5946                  |
| **Latitude**                | Latitude coordinate                          | 12.9716                  |
| **Timestamp1**              | Measurement date/time                        | 2025-08-12 14:30:00      |
| **Timestamp2** *(optional)* | End time of measurement                      | 2025-08-12 14:35:00      |
""")
    
    st.subheader("✨ Key Features")
    st.markdown("""
-  **Flexible Data Input** – Load the default CSV or upload your own dataset  
-  **Signal Strength Analysis** – Explore distributions and a quick ML benchmark  
-  **Coverage Map Visualization** – Interactive folium maps  
-  **Coverage Gap Detection** – Highlight problem zones  
-  **Reverse/Forward Geocoding** – Lat/Lon ⇄ Address  
-  **Wi-Fi Signal Strength (Windows)** – PC’s current Wi-Fi quality
""")

# --- TAB 2: Signal Strength Analysis ---
with tabs[1]:
    st.header(" Signal Strength Analysis")
    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)

    st.subheader("Signal Strength Distribution")
    plot_signal_distribution(df)
    
    st.subheader("Model Test on CSV")
    target_col = "Signal"
    label_encoders = {}
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le

    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    if len(X.columns) > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        st.write(f"### ✅ Model Accuracy: {acc*100:.2f}%")
        st.text("Classification Report:\n" + classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    else:
        st.info("Not enough non-target columns to train a quick classifier.")

# --- TAB 3: Coverage Map ---
with tabs[2]:
    st.header("Coverage Map")
    map_path = "map.html"

    df.to_csv("temp_for_map.csv", index=False)
    generate_coverage_map("temp_for_map.csv", map_path)

    if os.path.exists(map_path):
        st.components.v1.html(open(map_path, 'r', encoding="utf-8").read(), height=600, scrolling=False)
    else:
        st.error("Map file not found.")

# --- TAB 4: Coverage Gap Detection ---
with tabs[3]:
    st.header("Coverage Gap Detection")

    def generate_coverage_gap_map(current_df):
        # Filter for Poor/No Signal
        gap_df = current_df[current_df['Signal'] == 0]
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
                    <b>Signal:</b> {row['Signal']}
                """
            ).add_to(gap_cluster)

        folium.LayerControl(collapsed=False).add_to(m)

        map_path_local = "coverage_gap_map.html"
        m.save(map_path_local)
        return map_path_local

    # Generate and display the gap map
    gap_map_path = generate_coverage_gap_map(df)
    if gap_map_path:
        st.components.v1.html(open(gap_map_path, 'r', encoding="utf-8").read(), height=500)
    else:
        st.warning("No coverage gaps found in the current dataset.")

# --- TAB 5: Predict Signal Strength ---
with tabs[4]:
    st.header("Predict Signal Strength")
    st.markdown("""Predict signal strength for any location using latitude and longitude. Uses the uploaded dataset or default Australia data.  
                Best results within your dataset's coverage area.""")
    # Train and store artifacts (only once)
    if "pred_artifacts" not in st.session_state or st.session_state["pred_artifacts"] is None:
        model, feat_list, net_le, defaults = train_regression_model(df)
        st.session_state["pred_artifacts"] = {
            "model": model,
            "features": feat_list,
            "le": net_le,
            "defaults": defaults,
        }

    artifacts = st.session_state["pred_artifacts"]

    # User input: Latitude & Longitude only
    st.subheader("Enter Location Coordinates")
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitude", value=-33.8688, format="%.6f")
    with col2:
        lon = st.number_input("Longitude", value=151.2093, format="%.6f")

    if st.button("Predict"):
        if lat and lon:
            pred_value, category, color = predict_signal_strength_with_artifacts(lat, lon, artifacts)

            st.success(f"📍 Prediction for ({lat}, {lon}):")
            st.write(f"**Signal Strength:** {pred_value:.2f} ")
            st.write(f"**Category:** {category}")

            # Show color indicator
            st.markdown(
                f"<div style='padding:10px;border-radius:10px;background-color:{color};color:white;text-align:center;'>"
                f"Signal Category: {category}"
                "</div>",
                unsafe_allow_html=True,
            )

            # Map with prediction point
            m = folium.Map(location=[lat, lon], zoom_start=14)
            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                color=color,
                fill=True,
                fill_color=color,
                popup=f"Predicted: {pred_value:.2f}  ({category})"
            ).add_to(m)

            st.components.v1.html(m._repr_html_(), height=400)
        else:
            st.error("Please provide valid Latitude & Longitude.")

    st.subheader("Example Coordinates")
    st.markdown("""

        | Category       | Location              | Latitude Range       | Longitude Range     |
        |----------------|-----------------------|----------------------|---------------------|
        | **Coastal (Urban)** | Sydney (NSW)         | -33.8 → -33.7        | 151.0 → 151.3       |
        |                | Melbourne (VIC)       | -37.9 → -37.7        | 144.9 → 145.2       |
        | **Outback (Sparse)** | Alice Springs (NT)   | -23.8 → -23.5        | 133.7 → 134.0       |
        |                | WA Desert Area        | -25.5 → -25.0        | 120.0 → 121.0       |
        | **Suburban**   | Perth Suburbs (WA)    | -31.9 → -31.7        | 115.8 → 116.0       |
        |                | Adelaide Hills (SA)   | -35.0 → -34.9        | 138.7 → 139.0       |
        | **Rural / Forested** | Tasmania (near Hobart) | -42.9 → -42.7        | 147.2 → 147.5       |
        |                | N. Queensland (Cairns)| -17.0 → -16.8        | 145.7 → 146.0       |
        | **Highway / Transit** | Stuart Highway (NT)   | -18.0 → -17.5        | 133.0 → 133.5       |
        |                | Nullarbor Plain (SA/WA) | -31.5 → -31.0        | 129.0 → 129.5       |
        """)

# --- TAB 6: Geocoding Tools ---
with tabs[5]:
    st.header("📍 Reverse Geocoding (Lat/Lon → Address)")
    lat_input = st.number_input("Latitude", format="%.6f", key="rev_lat")
    lon_input = st.number_input("Longitude", format="%.6f", key="rev_lon")
    if st.button("Get Address", key="btn_get_address"):
        st.success(reverse_geocode(lat_input, lon_input))

    st.markdown("---")
    st.header("📍 Forward Geocoding (Address → Lat/Lon)")
    address_input = st.text_input("Enter location address:", "", key="forward_addr")
    if st.button("Get Coordinates", key="btn_get_coords"):
        if address_input.strip():
            geo_lat, geo_lon = geocode_address(address_input)
            if geo_lat is not None and geo_lon is not None:
                st.success(f"Latitude: {geo_lat}, Longitude: {geo_lon}")
                st.map(pd.DataFrame({"lat": [geo_lat], "lon": [geo_lon]}))
            else:
                st.error("Address not found.")
        else:
            st.warning("Please enter an address.")
    st.subheader("Example Coordinates")
    st.markdown("""

        | Category       | Location              | Latitude Range       | Longitude Range     |
        |----------------|-----------------------|----------------------|---------------------|
        | **Coastal (Urban)** | Sydney (NSW)         | -33.8 → -33.7        | 151.0 → 151.3       |
        |                | Melbourne (VIC)       | -37.9 → -37.7        | 144.9 → 145.2       |
        | **Outback (Sparse)** | Alice Springs (NT)   | -23.8 → -23.5        | 133.7 → 134.0       |
        |                | WA Desert Area        | -25.5 → -25.0        | 120.0 → 121.0       |
        | **Suburban**   | Perth Suburbs (WA)    | -31.9 → -31.7        | 115.8 → 116.0       |
        |                | Adelaide Hills (SA)   | -35.0 → -34.9        | 138.7 → 139.0       |
        | **Rural / Forested** | Tasmania (near Hobart) | -42.9 → -42.7        | 147.2 → 147.5       |
        |                | N. Queensland (Cairns)| -17.0 → -16.8        | 145.7 → 146.0       |
        | **Highway / Transit** | Stuart Highway (NT)   | -18.0 → -17.5        | 133.0 → 133.5       |
        |                | Nullarbor Plain (SA/WA) | -31.5 → -31.0        | 129.0 → 129.5       |
        """)



# --- TAB 7: Wi-Fi Strength (Windows) ---
with tabs[6]:
    st.header("Windows Wi-Fi Strength")
    st.info("Only works on Windows PC.")
    if st.button("Check Wi-Fi Strength", key="btn_wifi"):
        st.success(get_windows_wifi_strength())