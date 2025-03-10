import joblib
import numpy as np
import streamlit as st
import pandas as pd

# Streamlit UI Title
st.title("📊 Placement Status Prediction App")
st.write("Enter the candidate details to predict placement status.")

# File Upload Section
model_file = st.file_uploader("Upload Model File (pkl)", type=["pkl"])
encoders_file = st.file_uploader("Upload Label Encoders (zip or pkl)", type=["pkl", "zip"])
data_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if model_file and encoders_file and data_file:
    # Load dataset
    df = pd.read_csv(data_file)
    st.success("Dataset Loaded Successfully!")
    
    # Load model & encoders
    model = joblib.load(model_file)
    label_encoders = joblib.load(encoders_file)
    st.success("Model and Label Encoders Loaded!")
    
    # UI for user input
    input_data = {
        "CGPA": st.number_input("CGPA", min_value=float(df['CGPA'].min()), max_value=float(df['CGPA'].max())),
        "Internships": st.number_input("Internships", min_value=int(df['Internships'].min()), max_value=int(df['Internships'].max())),
        "Projects": st.number_input("Projects", min_value=int(df['Projects'].min()), max_value=int(df['Projects'].max())),
        "Workshops/Certifications": st.number_input("Workshops/Certifications", min_value=int(df['Workshops/Certifications'].min()), max_value=int(df['Workshops/Certifications'].max())),
        "AptitudeTestScore": st.number_input("Aptitude Test Score", min_value=int(df['AptitudeTestScore'].min()), max_value=int(df['AptitudeTestScore'].max())),
        "SoftSkillsRating": st.number_input("Soft Skills Rating", min_value=int(df['SoftSkillsRating'].min()), max_value=int(df['SoftSkillsRating'].max())),
        "ExtracurricularActivities": st.selectbox("Extracurricular Activities", df['ExtracurricularActivities'].unique()),
        "PlacementTraining": st.selectbox("Placement Training", df['PlacementTraining'].unique()),
        "SSC_Marks": st.number_input("SSC Marks", min_value=int(df['SSC_Marks'].min()), max_value=int(df['SSC_Marks'].max())),
        "HSC_Marks": st.number_input("HSC Marks", min_value=int(df['HSC_Marks'].min()), max_value=int(df['HSC_Marks'].max())),
    }

    # Preprocessing function
    def preprocess_input(data, label_encoders):
        data['ExtracurricularActivities'] = label_encoders['ExtracurricularActivities'].transform([data['ExtracurricularActivities']])[0]
        data['PlacementTraining'] = label_encoders['PlacementTraining'].transform([data['PlacementTraining']])[0]
        return np.array(list(data.values())).reshape(1, -1)

    if st.button("🔍 Predict Placement Status"):
        processed_data = preprocess_input(input_data, label_encoders)
        prediction = model.predict(processed_data)
        placement_status_label = label_encoders['PlacementStatus'].inverse_transform(prediction)[0]
        st.success(f"🎯 Predicted Placement Status: **{placement_status_label}**")

else:
    st.warning("Please upload all required files (Model, Encoders, and Dataset)")
