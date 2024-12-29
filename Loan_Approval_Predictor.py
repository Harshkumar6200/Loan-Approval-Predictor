import streamlit as st
import pickle
import numpy as np
import os

# Load the models
def load_model(filename):
    current_dir = os.path.dirname(__file__)  # Get the current script directory
    filepath = os.path.join(current_dir, filename)  # Construct the full file path
    with open(filepath, 'rb') as file:  # Use the dynamic filename
        return pickle.load(file)

# Load all models
try:
    rf = load_model("rf.pkl")  # Load Random Forest model
    bg = load_model("bg.pkl")  # Load Bagging Classifier model
    gb = load_model("gb.pkl")  # Load Gradient Boosting model
    Dt = load_model("Dt.pkl")  # Load Decision Tree model
except FileNotFoundError as e:
    st.error(f"Error: {e}. Ensure the model files are in the same directory as this script.")
    st.stop()

# Streamlit app setup
st.title("Loan Approval Prediction App")
st.header('Fill the Details to Generate the Predicted Loan Approval')

# Model selection
options = st.sidebar.selectbox(
    'Select ML Model', 
    ['Random Forest Classifier', 'Bagging Classifier', 'Gradient Boosting', 'Decision Tree Classifier']
)

# User input
no_of_dependents = st.slider('Number of dependents', 0, 5)
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
income_annum = st.slider('Income Per Annum', 200000, 9900000)
loan_amount = st.slider('Loan Amount', 300000, 39500000)
loan_term = st.slider('Loan Term (Years)', 2, 20)
cibil_score = st.slider('Cibil Score', 300, 900)
residential_assets_value = st.slider('Residential assets value', -100000, 30000000)
commercial_assets_value = st.slider('Commercial assets value', 0, 20000000)
luxury_assets_value = st.slider('Luxury assets value', 300000, 39200000)
bank_asset_value = st.slider('Bank asset value', 0, 15000000)

# Prediction button
if st.button('Predict'):
    try:
        # Convert categorical variables to numerical
        education = 1 if education == 'Graduate' else 0
        self_employed = 1 if self_employed == 'Yes' else 0

        # Create the test array with the correct number of features
        test = np.array([
            no_of_dependents, education, self_employed, income_annum, loan_amount, loan_term, cibil_score,
            residential_assets_value, commercial_assets_value, luxury_assets_value, bank_asset_value
        ]).reshape(1, -1)

        # Predict using the selected model
        if options == 'Random Forest Classifier':
            prediction = rf.predict(test)[0]
        elif options == 'Bagging Classifier':
            prediction = bg.predict(test)[0]
        elif options == 'Gradient Boosting':
            prediction = gb.predict(test)[0]
        else:  # 'Decision Tree Classifier'
            prediction = Dt.predict(test)[0]

        # Determine the loan status
        loan_status = 'Approved' if prediction == 1 else 'Rejected'

        # Display the prediction
        st.success(f'Your Loan Status is: {loan_status}')
    except Exception as e:
        st.error(f"An error occurred: {e}")
