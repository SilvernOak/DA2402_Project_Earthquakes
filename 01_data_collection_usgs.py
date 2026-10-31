import os
import requests
import pandas as pd
from datetime import date
from urllib.robotparser import RobotFileParser

# 1. Setup Data Folders & Metadata
os.makedirs("raw", exist_ok=True)
USER_AGENT = "DA2402_Project/1.0 (da25b015@smail.iitm.ac.in)"


# =====================================================================
# 2. CHECK ROBOTS.TXT & TERMS OF USE (Rubric Requirement)
# =====================================================================
# USGS API subdomain has no robots.txt, so we check the root domain.
robots_url = "https://www.usgs.gov/robots.txt"
resp_robots = requests.get(robots_url, headers={"User-Agent": USER_AGENT})

print(f"Robots.txt Status: {resp_robots.status_code}")
print("--- First 10 lines of robots.txt for the report ---")
print("\n".join(resp_robots.text.splitlines()[:10]))

# Check if our specific API endpoint is allowed
parser = RobotFileParser()
parser.parse(resp_robots.text.splitlines())
is_allowed = parser.can_fetch(USER_AGENT, "https://earthquake.usgs.gov/fdsnws/event/1/query")
print(f"\nAm I allowed to scrape the API? {is_allowed}")
print("Terms of Use: USGS data is U.S. Government Work (Public Domain / CC0).")

# =====================================================================
# 3. END-TO-END API CALL & SAVE RAW FILE (Rubric Requirement)
# =====================================================================
# We need M >= 6.0 from 2000 to 2025. Total events are ~3,875.
# Since the USGS limit is 20,000 per query, we can do this in ONE call.
api_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
params = {
    "format": "csv",
    "starttime": "2000-01-01T00:00:00",
    "endtime": "2025-12-31T23:59:59",
    "minmagnitude": 6.0,
    "orderby": "time-asc"
}

print(f"\nFetching data from: {api_url}")
resp_api = requests.get(api_url, params=params, headers={"User-Agent": USER_AGENT})
resp_api.raise_for_status()

# Save exactly as it came from the API (Never edit raw files!)
raw_filepath = f"raw/usgs_events_2000_2025.csv"
with open(raw_filepath, "wb") as f:
    f.write(resp_api.content)
    
print(f"Success! Saved raw data to {raw_filepath}")

# =====================================================================
# 4. RAW FILE INSPECTION (Rubric Requirement)
# =====================================================================
# Load the saved file to show shape, dtypes, and meaning of kept columns
df_raw = pd.read_csv(raw_filepath)

# We will drop station/instrument calibration columns during cleaning.
kept_cols = ["id", "time", "latitude", "longitude", "depth", "mag", "magType", "place", "type"]
df_kept = df_raw[kept_cols]

print("\n--- Raw Data Inspection ---")
print(f"Shape: {df_kept.shape}")
print(f"\nDtypes:\n{df_kept.dtypes}")

# Stating the meaning of every kept column for the rubric
column_meanings = {
    "id": "Unique string ID for the earthquake",
    "time": "UTC Timestamp (ISO-8601)",
    "latitude": "WGS84 Latitude (Decimal degrees)",
    "longitude": "WGS84 Longitude (Decimal degrees)",
    "depth": "Focal depth in kilometers",
    "mag": "Reported earthquake magnitude",
    "magType": "Magnitude scale used (e.g., mww, mb)",
    "place": "Unstructured text description of location",
    "type": "Event type (earthquake, nuclear explosion)"
}

print("\nMeaning of kept columns:")
for col, meaning in column_meanings.items():
    print(f" - {col}: {meaning}")