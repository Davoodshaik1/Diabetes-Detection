import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv')

# separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Streamlit App
st.title("Diabetes Prediction App")

# Add background image
st.markdown(
    """
    <style>
    body {
        background-image: url('https://gomohealth.com/wp-content/uploads/2019/08/Diabetes-Blood-Test-Gradient-Background.jpg'); /* Replace 'URL_TO_YOUR_IMAGE' with the actual URL of your image */
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Introduction text
st.write(
    "Welcome to the Diabetes Prediction App! This app uses a machine learning model to predict whether a person has diabetes "
    "based on various health-related features. Adjust the sliders on the sidebar to input your data and see the prediction."
)

# User guidance text
st.write(
    "### Instructions\n"
    "1. Use the sliders on the left to input your health-related features."
    "\n2. Click the 'Run Prediction' button to see the model's prediction."
    "\n3. The prediction will be displayed below."
    "\n4. The entire line will be highlighted in green if the model predicts diabetes, and in red if it predicts no diabetes."
)

# Input features
st.sidebar.header('User Input Features')
pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
glucose = st.sidebar.slider('Glucose', 0, 199, 117)
blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
insulin = st.sidebar.slider('Insulin', 0, 846, 30)
bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
age = st.sidebar.slider('Age', 21, 81, 29)

# User input data
user_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]).reshape(1, -1)

# Make predictions
prediction = classifier.predict(user_data)

# Display prediction with highlighted line
st.subheader('Prediction:')
if prediction[0] == 0:
    st.markdown('<div style="color: green;">The model predicts that the person does not have diabetes.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="color: red;">The model predicts that the person has diabetes.</div>', unsafe_allow_html=True)
