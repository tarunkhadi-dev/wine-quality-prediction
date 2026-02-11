import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.title("Wine Quality Prediction 🍷")

# Load dataset
data = pd.read_csv("data/winequality-red.csv")
data['quality_label'] = data['quality'].apply(lambda x: 1 if x >= 7 else 0)

X = data.drop(['quality', 'quality_label'], axis=1)
y = data['quality_label']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

st.write("Enter wine details:")

fixed_acidity = st.number_input("Fixed Acidity", 0.0)
volatile_acidity = st.number_input("Volatile Acidity", 0.0)
citric_acid = st.number_input("Citric Acid", 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.0)
chlorides = st.number_input("Chlorides", 0.0)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0)
density = st.number_input("Density", 0.0)
pH = st.number_input("pH", 0.0)
sulphates = st.number_input("Sulphates", 0.0)
alcohol = st.number_input("Alcohol", 0.0)

if st.button("Predict"):
    input_data = [[
        fixed_acidity, volatile_acidity, citric_acid,
        residual_sugar, chlorides, free_sulfur_dioxide,
        total_sulfur_dioxide, density, pH,
        sulphates, alcohol
    ]]
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("Good Quality Wine 🍷")
    else:
        st.error("Bad Quality Wine ❌")
