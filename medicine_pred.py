import streamlit as st
import pandas as pd
import numpy as np
import easyocr
import cv2
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from langchain_ollama import ChatOllama
from PIL import Image

# ---------------------------------
# PAGE CONFIG (ONLY ONCE)
# ---------------------------------
st.set_page_config(page_title="Medicine Overdose AI System", layout="wide")

st.title("💊 Medicine Overdose Prediction System")

# ---------------------------------
# CREATE TWO TABS
# ---------------------------------
tab1, tab2 = st.tabs([
    "Manual Dataset Prediction",
    "OCR Prescription Prediction"
])
st.sidebar.header("Patient Information")
age = st.sidebar.number_input("Patient Age", 1, 120, 30, key="age_input")
weight = st.sidebar.number_input("Patient Weight (kg)", 2, 200, 70, key="weight_input")

# ====================================================================================
# ============================ TAB 1 : MIDDLE.PY LOGIC ===============================
# ====================================================================================
with tab1:

    st.header("Manual Dataset Based Prediction")

    uploaded_file = st.file_uploader(
        "Upload Medicine Dataset (CSV)",
        type=["csv"],
        key="dataset_upload"
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully")
        #st.dataframe(df.head())

        encoders = {}

        for col in ["medicine_name", "overdose_risk"]:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le

        X = df[["medicine_name", "dosage_mg", "frequency_per_day"]]
        y = df["overdose_risk"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)

        st.success(f"Model trained | Accuracy: {accuracy:.2f}")

        st.subheader("Enter Prescription Details")

        medicine = st.selectbox(
            "Medicine Name",
            encoders["medicine_name"].classes_
        )

        dosage = st.number_input("Dosage (mg)", min_value=0, step=50)
        frequency = st.number_input("Frequency per Day", min_value=0, step=1)
        st.subheader("Are you under any other medications?")
        col1, col2 = st.columns([1, 10])
        with col1:
            y = st.button("Yes")
        with col2:
            n = st.button("No")
        if y:
            medicine = st.selectbox(
                "Medicine Name",
                encoders["medicine_name"].classes_,
                key="medicine_select_1"
            )

        if st.button("Predict Overdose Risk", key="manual_predict"):

            input_data = pd.DataFrame([[
                encoders["medicine_name"].transform([medicine])[0],
                dosage,
                frequency
            ]], columns=X.columns)

            prediction = model.predict(input_data)[0]
            risk = encoders["overdose_risk"].inverse_transform([prediction])[0]

            if risk.lower() == "high":
                st.error("⚠️ HIGH RISK")
            elif risk.lower() == "medium":
                st.warning("⚠️ MEDIUM RISK")
            else:
                st.success("✅ LOW RISK")

            # Ollama Explanation
            llm = ChatOllama(model="llama3", temperature=0)

            prompt = f"""
            A patient is prescribed:
            Medicine: {medicine}
            Dosage: {dosage} mg
            Frequency: {frequency} per day
            - Age: {age}
            Risk: {risk}

            Explain reason, safe limits and precautions.
            """

            explanation = llm.invoke(prompt)
            st.subheader("AI Explanation")
            st.write(explanation.content)

# ====================================================================================
# ============================== TAB 2 : OCR.PY LOGIC ================================
# ====================================================================================
with tab2:

    st.header("Prescription Image OCR Prediction")

    @st.cache_resource
    def train_model():
        data = {
            'age': [25, 70, 45, 10, 80, 30, 5, 55, 12, 65, 90, 4, 50, 85],
            'weight': [70, 60, 85, 30, 55, 75, 18, 90, 40, 70, 50, 15, 100, 58],
            'dosage_mg': [500, 2000, 1000, 250, 1500, 500, 100, 3000, 1000, 500, 1200, 500, 1500, 2000],
            'frequency_per_day': [2, 4, 2, 1, 3, 2, 1, 6, 4, 1, 4, 3, 2, 3],
            'is_overdose': [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]
        }

        df = pd.DataFrame(data)
        X = df.drop('is_overdose', axis=1)
        y = df['is_overdose']

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    ml_model = train_model()
    reader = easyocr.Reader(['en'])

    

    uploaded_image = st.file_uploader(
        "Upload Prescription Image",
        type=["jpg", "jpeg", "png"],
        key="image_upload"
    )

    if uploaded_image:

        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Prescription", width=400)

        img_array = np.array(image)
        results = reader.readtext(img_array, detail=0)
        full_text = " ".join(results)

        dosages = re.findall(r'(\d+(?:\.\d+)?)\s*(mg|g)', full_text.lower())

        dosage_values_mg = []
        for value, unit in dosages:
            value = float(value)
            if unit == 'g':
                value *= 1000
            dosage_values_mg.append(value)

        extracted_dosage = int(max(dosage_values_mg)) if dosage_values_mg else 0

        freq = 4 if any(word in full_text.lower() for word in ["times", "day", "qds"]) else 2

        features = np.array([[age, weight, extracted_dosage, freq]])
        prob = ml_model.predict_proba(features)[0][1]
        risk_score = prob * 100

        st.subheader("Extracted Details")
        st.write(f"Detected Dosage: {extracted_dosage} mg")
        st.write(f"Estimated Frequency: {freq} times/day")

        st.subheader("Risk Analysis")

        if risk_score > 60:
            st.error("HIGH RISK PRECAUTIONS")
        elif risk_score > 30:
            st.warning("MODERATE RISK PRECAUTIONS")
        else:
            st.success("LOW RISK PRECAUTIONS")

        st.progress(int(risk_score))

    else:
        st.info("Upload a prescription image to start.")