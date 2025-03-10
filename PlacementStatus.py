import joblib
import numpy as np
import streamlit as st
import pandas as pd
# Load dataset
import streamlit as st
import pandas as pd

# Load the dataset
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)  # Read uploaded file
    st.write(df.head())  # Display first few rows
else:
    st.warning("Please upload a CSV file.")  # Show warning
    df = None  # Set df to None when no file is uploaded

# Ensure df exists before using it
if df is not None:
    st.number_input("CGPA", min_value=int(df['CGPA'].min()), max_value=int(df['CGPA'].max()))
else:
    st.error("Error: No data available. Please upload a valid CSV file.")



# Function to load the saved model and label encoders
def load_model():
    model = uploaded_file = st.file_uploader("Upload Model File (.pkl)", type="pkl")
    if uploaded_file is not None:
        model = joblib.load(uploaded_file)  # Load the uploaded model
        st.write("Model loaded successfully!")
        import joblib
import os

def load_model():
    model_path = "AdaBoostClassifier_model.pkl"  # Adjust path if needed
    encoder_path = "label_encoders.pkl"  # If using label encoders

    # Initialize label_encoders
    label_encoders = None  

    # Check if model file exists before loading
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Error: Model file not found at {model_path}")

    # Check if label encoders exist before loading
    if os.path.exists(encoder_path):
        label_encoders = joblib.load(encoder_path)  
    else:
        print("Warning: Label encoders not found. Proceeding without them.")

    return model, label_encoders  # Always returns something

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

