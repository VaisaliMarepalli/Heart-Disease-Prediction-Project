#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier


# In[2]:


model = joblib.load("heart_disease_model.pkl")


# In[3]:


st.set_page_config(page_title="Heart Disease Prediction App", page_icon="❤️", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below and get a quick prediction.")


# In[4]:


# 2. Create Input Fields


# In[5]:


col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=100, value=None)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type (cp)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=None)
    chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=None)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    restecg = st.selectbox("Resting ECG Results (restecg)", options=[0, 1, 2])
    thalach = st.number_input("Maximum Heart Rate (thalach)", min_value=70, max_value=220, value=None)
    exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=6.0, value=None, step=0.1)
    slope = st.selectbox("Slope of ST Segment (slope)", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (ca)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", options=[0, 1, 2, 3])


# In[6]:


# 3. Make Prediction


# In[7]:


if st.button("🔍 Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                            exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ The model predicts **Heart Disease (Positive)**. Please consult a doctor.")
    else:
        st.success("✅ The model predicts **No Heart Disease (Negative)**. You seem safe!")

st.write("---")
st.caption("Powered by Machine Learning | Built with Streamlit")


# In[ ]:




