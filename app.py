import streamlit as st
import joblib
import os
import pandas as pd
import pickle

# -----------------------------
# Load district→state mapping
# -----------------------------
district_state_map = pickle.load(open("district_state_map.pkl", "rb"))

# -----------------------------
# 2022 Environmental Values
# -----------------------------
ENV_2022 = {
    "Uttar Pradesh":  {"NDVI": 4798.719532, "Rainfall": 877.0945416, "Temperature": 28.66469342},
    "Punjab":         {"NDVI": 5846.595107, "Rainfall": 710.1282511, "Temperature": 29.40444777},
    "Haryana":        {"NDVI": 4310.983064, "Rainfall": 565.8587185, "Temperature": 30.1846854}
}

MODEL_DIR = "crop_models"

# -----------------------------
# Load all crop models
# -----------------------------
@st.cache_resource
def load_all_models():
    models = {}
    for f in os.listdir(MODEL_DIR):
        if f.endswith(".pkl"):
            crop = f[:-4]
            models[crop] = joblib.load(os.path.join(MODEL_DIR, f))
    return models

models = load_all_models()

# -----------------------------
# UI
# -----------------------------
st.title("🌾 Crop Recommendation (State-based, 2022 Conditions)")
st.write("Enter State & District to get best crops for your region.")

state = st.selectbox("Select State", ["Uttar Pradesh", "Punjab", "Haryana"])
district = st.text_input("Enter District Name (case-sensitive)")

if st.button("Recommend Crops"):
    if district == "":
        st.error("❌ Please enter a district.")
    else:
        # Get ENV for selected state
        env = ENV_2022[state]

        # Pipeline requires this exact structure
        input_row = pd.DataFrame([{
            "State_x": state,
            "District": district,
            "NDVI": env["NDVI"],
            "Rainfall": env["Rainfall"],
            "Temperature": env["Temperature"],
            "Area": 1
        }])

        preds = []

        # Run prediction for all crop models
        for crop, model in models.items():
            try:
                pred = model.predict(input_row)[0]
                preds.append((crop, pred))
            except:
                pass

        if len(preds) == 0:
            st.error("⚠ No predictions generated. District may be incorrect.")
        else:
            preds = sorted(preds, key=lambda x: x[1], reverse=True)

            st.subheader("🌱 Top 3 Crops")
            for crop, val in preds[:3]:
                st.write(f"**{crop}** → {val:.2f}")

            st.subheader("📊 All Predictions")
            st.dataframe(pd.DataFrame(preds, columns=["Crop", "Predicted Yield"]))
