import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# Laod the trained model
# -------------------------------
# model = pickle.load(open('titanic_model.pkl','rb'))

with open('titanic_model.pkl','rb') as file:
    model = pickle.load(file)

# -------------------------------
# App Title
# -------------------------------
st.title('🚢 Titanic Survival Prediction App')
st.markdown("This Streamlit application predicts whether a Titanic passenger would survive based on the provided information.")
st.write('Enter passenger details to predict survival probability.')

# -------------------------------
# Enter Inputs
# -------------------------------
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.radio("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80, 29)
fare = st.slider("Fare", 0.0, 500.0, 32.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode user inputs exactly like training
sex = 1 if sex == "Male" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Create DataFrame in same order as training features
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [0],
    'Parch': [0],
    'Fare': [fare],
    'encoded_male': [sex],
    'encoded_Q': [embarked_Q],
    'encoded_S': [embarked_S]
})

# -------------------------------
# Make Prediction
# -------------------------------
if st.button("Predict Survival"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    if pred == 1:
        st.success(f" Passenger would survive! (Probability: {prob:.2f})")
    else:
        st.error(f" Passenger would not survive. (Probability: {prob:.2f})")
