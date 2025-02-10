import streamlit as st
import joblib
import numpy as np
import os

# Check if the model file exists
if os.path.exists("iris_model.pkl"):
    model = joblib.load("iris_model.pkl")
else:
    st.error("Error: Model file 'iris_model.pkl' not found. Please upload it to GitHub.")

# Streamlit App UI
st.title("🌸 Iris Flower Classifier")
st.write("Enter the flower's features to predict its species")

# Input fields for flower features
sepal_length = st.number_input("Sepal Length", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.0, max_value=10.0, step=0.1)

# Button to make predictions
if st.button("Predict"):
    if 'model' in globals():
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)
        species = ["Setosa", "Versicolor", "Virginica"]
        st.success(f"The predicted species is: {species[prediction[0]]}")
    else:
        st.error("Model is not loaded. Please check the model file.")
