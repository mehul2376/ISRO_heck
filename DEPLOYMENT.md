# Streamlit Cloud Deployment Guide

This document outlines the exact steps to deploy the EarthMatrix application (Surface AQI & HCHO Hotspot Analysis) to Streamlit Cloud.

## Deployment Steps

1. **Go to Streamlit Cloud**
   Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.

2. **Deploy a New App**
   Click the **"New app"** button in the top right corner.

3. **Configure the App Settings**
   Fill out the deployment form with the following exact details:
   * **Repository**: `mehul2376/ISRO_heck`
   * **Branch**: `main`
   * **Main file path**: `app.py`
   * **App URL**: `earthmatrix-aqi-hcho`

   > **Important:** Do not put `earthmatrix-aqi-hcho` in the "Main file path" field. The Main file path MUST be `app.py`.

4. **Click Deploy**
   Click the **Deploy!** button. Streamlit will now read your `requirements.txt` and install the necessary dependencies. The app will be available in 2-3 minutes at `https://earthmatrix-aqi-hcho.streamlit.app`.

---

## Auto-Redeploy
After the app is connected to Streamlit Cloud, every new push to the `main` branch will **automatically redeploy** the app. Please keep all future updates pushed to the `main` branch so the live app stays up to date.

---

## Verification Checklist
Once the app is deployed, verify the following:
- [ ] App opens successfully without `ModuleNotFoundError`.
- [ ] Data Explorer and Overview tabs load correctly.
- [ ] AQI Map loads the map and mock hotspots without error.
- [ ] HCHO Hotspots tab loads without error.
- [ ] Fire & Wind Transport tab loads successfully.
- [ ] Model Performance (Time Series) tab loads successfully.
- [ ] The app successfully uses Mock/Demo Mode by default (No Google Earth Engine auth errors).
- [ ] The TensorFlow fallback message displays correctly on the Time Series tab: *"TensorFlow model architecture available locally; cloud demo uses cached predictions."*

---

## Troubleshooting

1. **`ModuleNotFoundError: No module named 'tensorflow'`**
   - **Fix:** TensorFlow is purposefully excluded from `requirements.txt` because it is too heavy for the free Streamlit Cloud tier and causes memory/build timeouts. The app contains safe `try/except` fallbacks and will run using cached predictions. Ensure `tensorflow` remains removed from `requirements.txt`.

2. **Google Earth Engine Authentication Error**
   - **Fix:** Ensure the app's sidebar has "Mock Mode" selected by default. The `gee_utils.py` module no longer tries to authenticate automatically unless Live GEE mode is actively triggered by the user with the proper secrets configured.

3. **Missing File / Path Errors**
   - **Fix:** Ensure the GitHub repository has the correct structure with `app.py` at the root and pages inside the `pages/` directory.

4. **Duplicate Widget Key Errors**
   - **Fix:** Ensure no two Streamlit inputs (like buttons or text inputs) share the exact same label without explicit unique `key` parameters.
