import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from PIL import Image


st.set_page_config(
    page_title="Insurance Cost Prediction",
    page_icon=":moneybag:",
    layout="centered"
)

model_filename = 'insurance_cost_model_gb.joblib'

data_path = 'insurance.csv'
insurance_data = pd.read_csv(data_path)



# Load the model for predictions
model = joblib.load(model_filename)

# App title and description
st.title("Insurance Cost Prediction App")
st.markdown("This app predicts medical insurance costs based on user input and provides insights with interactive data visualizations.")
background_image = Image.open('img.webp')
st.image(background_image, caption="Make informed decisions based on real data insights!", use_column_width=True)

# Input Section
st.header("Input Personal Details")

col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    sex = st.selectbox("Sex", options=['male', 'female'])

with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    smoker = st.selectbox("Smoker", options=['yes', 'no'])

with col3:
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0, step=1)
    region = st.selectbox("Region", options=['northeast', 'southeast', 'southwest', 'northwest'])

# Predict button
if st.button("Predict Insurance Cost"):
    # Prepare input data for prediction
    input_data = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                              columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
    prediction = model.predict(input_data)[0]

    st.write(f"Predicted Insurance Cost: ${prediction:.2f}")

    # Recommendations based on input data
    st.header("Recommendations to Reduce Insurance Cost")
    
    recommendations = []
    
    # Recommendation for BMI
    if bmi > 24.9:
        recommendations.append("Consider working towards a healthier BMI through balanced diet and regular exercise.")
        
    # Recommendation for smokers
    if smoker == 'yes':
        recommendations.append("Quitting smoking can significantly reduce your insurance costs.")

    # Recommendation for age
    if age > 50:
        recommendations.append("Explore insurance plans that offer senior discounts to help reduce costs.")
        
    # Children factor
    if children > 0:
        recommendations.append("Family insurance plans might be more cost-effective if you have multiple dependents.")

    # Display recommendations
    if prediction < 5000:
        st.balloons()
        recommendations.append(f"🎉 Amazing! Your insurance charges are quite low at ${prediction:.2f}!")
        recommendations.append("Keep up the great work with your health and lifestyle choices!")
    elif 5000 <= prediction <= 15000:
        st.snow()
        recommendations.append(f"👍 Your insurance charges are moderate at ${prediction:.2f}.")
        recommendations.append("You're doing well, but there might be a few areas for improvement to reduce costs.")
    else:
        st.warning(f"⚠️ Your insurance charges are higher than average at ${prediction:.2f}.")
        recommendations.append("Consider lifestyle changes, such as regular exercise or quitting smoking, to potentially lower costs.")
    if recommendations:
        for rec in recommendations:
            st.write("- " + rec)

# Data Visualization Section

# Visualization 1: Distribution of charges
st.header("Distribution of Insurance Charges")
fig = px.histogram(insurance_data, x="charges", nbins=50, title="Distribution of Insurance Charges", marginal="box")
st.plotly_chart(fig, use_container_width=True)

# Visualization 2: BMI vs Charges with hover effect
st.subheader("Impact of BMI on Insurance Charges")
smoker_filter = st.selectbox("Filter by Smoker Status", options=['all', 'yes', 'no'])
filtered_data = insurance_data if smoker_filter == 'all' else insurance_data[insurance_data['smoker'] == smoker_filter]

fig = px.scatter(filtered_data, x="bmi", y="charges", color="smoker",
                 title=f"BMI vs Insurance Charges (Smoker: {smoker_filter})",
                 hover_data=['age', 'children', 'region'])
st.plotly_chart(fig, use_container_width=True)

# Visualization 3: Age vs Charges with slider for age range and hover effect
st.subheader("Impact of Age on Insurance Charges")
age_min, age_max = st.slider("Select Age Range", 18, 100, (18, 100))
age_filtered_data = insurance_data[(insurance_data['age'] >= age_min) & (insurance_data['age'] <= age_max)]

fig = px.scatter(age_filtered_data, x="age", y="charges", color="smoker",
                 title=f"Age vs Insurance Charges (Age Range: {age_min}-{age_max})",
                 hover_data=['bmi', 'children', 'region'])
st.plotly_chart(fig, use_container_width=True)

# Visualization 4: Average charges by region with hover effect
st.subheader("Average Insurance Charges by Region")
fig = px.bar(insurance_data, x="region", y="charges", color="region", 
             title="Average Insurance Charges by Region", 
             hover_data=['charges'])
st.plotly_chart(fig, use_container_width=True)
