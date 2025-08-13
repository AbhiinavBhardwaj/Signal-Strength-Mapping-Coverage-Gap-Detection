# 📡 Signal Strength Mapping & Coverage Gap Detection

An interactive **Streamlit** web application to **visualize, analyze, and detect mobile network coverage gaps** using geospatial and signal strength data.

---

## ✨ Features
- **Flexible Data Input** – Upload your own CSV or use the default dataset
- **Signal Strength Analysis** – Charts, model predictions, and category distribution
- **Coverage Map** – Interactive folium map with markers colored by signal quality
- **Coverage Gap Detection** – Identify areas with poor or no signal
- **Geocoding Tools** – Forward and reverse geocoding using OpenStreetMap's Nominatim
- **Windows Wi-Fi Strength** – Check current Wi-Fi signal quality (Windows only)

---

## 📂 Required CSV Format
| Column Name                 | Description                                  | Example Value            |
|-----------------------------|----------------------------------------------|--------------------------|
| **Network_Type**            | Mobile network technology                    | 4G, 5G                   |
| **MCC**                     | Mobile Country Code                          | 404 (India)              |
| **MNC**                     | Mobile Network Code                          | 10 (Airtel)              |
| **LAC**                     | Location Area Code                           | 23456                    |
| **Cell_ID**                 | Cell tower identifier                        | 12345678                 |
| **Signal**                  | Signal strength                              | 0, 50, 80, 100           |
| **Longitude**               | Longitude coordinate                         | 77.5946                  |
| **Latitude**                | Latitude coordinate                          | 12.9716                  |
| **Timestamp1**              | Measurement date/time                        | 2025-08-12 14:30:00      |
| **Timestamp2** *(optional)* | End time of measurement                      | 2025-08-12 14:35:00      |

---

## 🚀 Installation & Running Locally
1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/signal-strength-mapping.git
   cd signal-strength-mapping

