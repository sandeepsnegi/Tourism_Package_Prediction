import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "package_purchase_prediction_model_v1.joblib")
model = joblib.load(model_path)

st.title("Package Purchase Prediction App")
st.write("""
This application predicts the likelihood of a Package Purchase based on its customer parameters.
Enter the customer data below to get a prediction.
""")

# Define input fields for user to enter data
Age                         = st.number_input("Age", 18, 80, 37, 1)
TypeofContact               = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier                    = st.number_input("City Tier", 1, 3, 1, 1)
DurationOfPitch             = st.number_input("Duration of Pitch (minutes)", 1, 200, 5, 1)
Occupation                  = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
Gender                      = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting      = st.number_input("Number of Persons Visiting", min_value=0)
NumberOfFollowups           = st.number_input("Number of Follow-ups", min_value=0)    
ProductPitched              = st.selectbox("Product Pitched", ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar       = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])
MaritalStatus               = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips               = st.number_input("Number of Trips", min_value=0)
Passport                    = st.number_input("Passport", 0, 1, 0, 1)
PitchSatisfactionScore      = st.number_input("Pitch Satisfaction Score", 1, 5, 3, 1)
OwnCar                      = st.number_input("Own Car", 0, 1, 0, 1)
NumberOfChildrenVisiting    = st.number_input("Number of Children Visiting", min_value=0)
Designation                 = st.selectbox("Designation", ["Manager", "Executive", "Senior Manager", "AVP", "VP"])
MonthlyIncome               = st.number_input("Monthly Income", min_value=0)

# Create a DataFrame from the user input
input_data = pd.DataFrame([{
    "Age": Age,
    "TypeofContact": TypeofContact,
    "CityTier": CityTier,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

# Make prediction when the button is clicked
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Package Purchased" if prediction == 1 else "Package Not Purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
