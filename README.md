# 🌾 Crop Yield Prediction & Crop Suitability Recommendation System

## 📌 Project Overview

This project builds a machine learning–based crop recommendation system for the Kharif season across three Indian states:

- Haryana  
- Uttar Pradesh  
- Punjab  

The system predicts crop yield using environmental factors and ranks crops from highest to lowest predicted yield for a selected district.

It helps identify which crop is best suited (yield-wise) for given Kharif environmental conditions.

---

## 📊 Data Sources

### 🛰 Environmental Data (Google Earth Engine)

Collected district-level Kharif season data (last 10 years):

- NDVI (Normalized Difference Vegetation Index)  
- Average Temperature  
- Rainfall  

States covered:

- Haryana  
- Uttar Pradesh  
- Punjab  

---

### 🌾 Crop Yield Data (Government Source)

Dataset includes:

- 55 crops  
- District-wise area of cultivation  
- Production quantity  
- Yield data  
- 10 years of Kharif season records  

---

## 🧹 Data Processing

- Merged environmental and crop datasets  
- Cleaned and standardized district names  
- Filtered for Kharif season only  

Final dataset features:

- NDVI  
- Rainfall  
- Average Temperature  
- District  
- State  
- Year  
- Crop  
- Yield (Target Variable)  

---

## 🤖 Model Architecture

- Model Used: LightGBM Regressor  
- Total Models Trained: 55 (one per crop)  
- Target Variable: Yield  

Each crop has a separate model trained using historical environmental and yield data.

---

## 💻 Application Logic

A local Streamlit-based web application was built.

### User Input

- State  
- District  

### Backend Logic

- Uses fixed Kharif 2022 environmental values (NDVI, rainfall, temperature)  
- Runs all 55 trained models  
- Predicts yield for each crop  
- Sorts crops from highest to lowest predicted yield  

### Output

A ranked list of crops best suited (yield-wise) for the selected district.

---

## 🚀 How to Run Locally

### 1. Clone the Repository

git clone https://github.com/AksharGupta1/Crop_Yield_Prediction.git  
cd Crop_Yield_Prediction  

### 2. Install Requirements

pip install -r requirements.txt  

### 3. Run the Application

streamlit run app.py  

The app will open automatically in your browser.

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- LightGBM  
- NumPy  
- Pandas  
- Scikit-learn  
- Joblib / Pickle  
- Google Earth Engine  

---

## 📈 Key Highlights

- Multi-model architecture (55 independent crop models)  
- District-level environmental integration  
- Practical agricultural decision-support use case  
- Remote sensing + government data integration  
- Scalable design for future deployment  

---

## 🔮 Future Improvements

- Extend to Rabi and Zaid seasons  
- Add crop rotation recommendations  
- Integrate weather forecast APIs  
- Include market price data for profitability ranking  
- Automate real-time NDVI extraction  
- Expand to additional Indian states  
- Deploy as a cloud-hosted decision-support platform  

---

## 🎯 Impact

This system demonstrates how remote sensing data and machine learning can be combined to support:

- Smarter crop planning  
- Yield optimization  
- Data-driven agricultural decisions  
- Scalable agri-tech solutions  
