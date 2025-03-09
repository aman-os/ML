import joblib
import numpy as np
import streamlit as st
import pandas as pd
# Load dataset
df = pd.read_csv(r"C:\Users\amanc\Git\ML\placedata v2.0 synthetic.csv")

# Function to load the saved model and label encoders
def load_model():
    model = joblib.load(r"C:\Users\amanc\PYTHON JUP\ML\Project\AdaBoostClassifier_model.pkl")
    label_encoders = {
        'ExtracurricularActivities': joblib.load(r"C:\Users\amanc\PYTHON JUP\ML\Project\L1"),
        'PlacementTraining': joblib.load(r"C:\Users\amanc\PYTHON JUP\ML\Project\L2"),
        'PlacementStatus': joblib.load(r"C:\Users\amanc\PYTHON JUP\ML\Project\L3"),
    }
    return model, label_encoders

# Preprocessing function
def preprocess_input(data, label_encoders):
    data['ExtracurricularActivities'] = label_encoders['ExtracurricularActivities'].transform([data['ExtracurricularActivities']])[0]
    data['PlacementTraining'] = label_encoders['PlacementTraining'].transform([data['PlacementTraining']])[0]
    return np.array(list(data.values())).reshape(1, -1)

# Load model and encoders
model, label_encoders = load_model()

# Streamlit UI
st.title("Placement Status Prediction")
st.write("Enter the details of the candidate to predict placement status.")

input_data = {
    "CGPA": st.number_input("CGPA", min_value=int(df['CGPA'].min()), max_value=int(df['CGPA'].max())),
    "Internships": st.number_input("Internships", min_value=int(df['Internships'].min()), max_value=int(df['Internships'].max())),
    "Projects": st.number_input("Projects", min_value=int(df['Projects'].min()), max_value=int(df['Projects'].max())),
    "Workshops/Certifications": st.number_input("Workshops/Certifications",min_value=int(df['Workshops/Certifications'].min()),max_value=int(df['Workshops/Certifications'].max())),
    "AptitudeTestScore": st.number_input("AptitudeTestScore", min_value=int(df['AptitudeTestScore'].min()), max_value=int(df['AptitudeTestScore'].max())),
    "SoftSkillsRating": st.number_input("SoftSkillsRating", min_value=int(df['SoftSkillsRating'].min()), max_value=int(df['SoftSkillsRating'].max())),
    "ExtracurricularActivities": st.selectbox("Extracurricular Activities", df['ExtracurricularActivities'].unique()),
    "PlacementTraining": st.selectbox("Placement Training", df['PlacementTraining'].unique()),
    "SSC_Marks": st.number_input("SSC_Marks", min_value=int(df['SSC_Marks'].min()), max_value=int(df['SSC_Marks'].max())),
    "HSC_Marks": st.number_input("HSC_Marks", min_value=int(df['HSC_Marks'].min()), max_value=int(df['HSC_Marks'].max())),
}

if st.button("Predict Placement Status"):
    processed_data = preprocess_input(input_data, label_encoders)
    prediction = model.predict(processed_data)
    
    # Convert numeric prediction back to label if necessary
    placement_status_label = label_encoders['PlacementStatus'].inverse_transform(prediction)[0]
    
    st.write(f"The predicted Placement Status of the candidate is: **{placement_status_label}**")
