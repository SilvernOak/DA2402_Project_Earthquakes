# DA2402_Project_Earthquakes

# 🌍 Quake: Earthquake Mortality, Magnitude, and Income (2000–2025)
**DA2402 · Data Curation Project**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Dash](https://img.shields.io/badge/Dash-Plotly-008080.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Curation-150458.svg)

## 📌 Project Overview
Earthquakes are a purely geological hazard, but the resulting human disaster is heavily influenced by socio-economic factors. This project features an end-to-end data curation pipeline and an interactive Plotly Dash application designed to answer a single core question: **After allowing for physical constraints (magnitude and depth), how much does a country's income explain earthquake deaths?**

**Key Finding:** Using a Two-Part Hurdle Model (Logistic + OLS regression), the data reveals that holding magnitude and depth constant, a **10x increase in a country's GDP per capita yields roughly a 76% reduction in expected fatalities.**

## 📊 Data Sources
Data was collected, merged, and reconciled from four sources spanning 2000 to 2025:
1. **[USGS ComCat API](https://earthquake.usgs.gov/fdsnws/event/1/)**: Base geological data for all $M6.0+$ mainshocks.
2. **[NOAA NCEI Hazel API](https://www.ngdc.noaa.gov/hazel/view/hazards/earthquake/search)**: Primary fatality counts.
3. **[Wikipedia MediaWiki](https://en.wikipedia.org/wiki/Lists_of_earthquakes)**: HTML-parsed tables and text for fatality data augmentation and cross-validation.
4. **[World Bank WDI API](https://data.worldbank.org/)**: Country-level economic indicators (GDP per capita PPP, Population Density).

## 🗂️ Repository Structure
This repository strictly follows data engineering best practices, ensuring raw data immutability and complete reproducibility.

```text
DA2402_Project_Earthquakes/
├── raw/                     # Untouched raw API JSONs and scraped CSVs
├── interim/                 # Scraping logs, robots.txt files, and partial cleaning outputs
├── clean/                   
│   ├── earthquakes_master_2000_2025_2026-10-28.parquet  # Final columnar dataset
│   └── earthquakes_master_2026-10-28.db                 # Parameterized SQLite backend
├── notebooks/
│   ├── 01_data_collection_*.ipynb  # Scripts extracting data from APIs and HTML
│   ├── 02_data_cleaning_*.ipynb    # Source-specific cleaning, typing, and imputation
│   ├── 03_complete_merger.ipynb    # Spatiotemporal clustering and hierarchical merging
│   ├── 04_visualisation.ipynb      # Static EDA and figure generation
│   └── dashboard.py                # Main Plotly Dash application
├── cleaning_log.csv
├── data_dictionary.csv
└── report.pdf
