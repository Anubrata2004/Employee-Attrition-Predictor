import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Load model, scaler, features
model = joblib.load('models/xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')
feature_names = joblib.load('models/feature_names.pkl')

st.set_page_config(page_title="Employee Attrition Predictor", 
                   page_icon="👥", layout="wide")

st.title("👥 Employee Attrition Predictor")
st.markdown("Enter employee details below to predict attrition risk.")

# ── Sidebar Inputs ──
st.sidebar.header("Employee Details")

age = st.sidebar.slider("Age", 18, 60, 30)
monthly_income = st.sidebar.number_input("Monthly Income (₹)", 1000, 20000, 5000)
distance = st.sidebar.slider("Distance From Home (km)", 1, 30, 10)
years_company = st.sidebar.slider("Years at Company", 0, 40, 3)
years_role = st.sidebar.slider("Years in Current Role", 0, 18, 2)
years_manager = st.sidebar.slider("Years with Current Manager", 0, 17, 2)
years_promoted = st.sidebar.slider("Years Since Last Promotion", 0, 15, 1)
total_working = st.sidebar.slider("Total Working Years", 0, 40, 5)
num_companies = st.sidebar.slider("Num Companies Worked", 0, 9, 2)
training = st.sidebar.slider("Training Times Last Year", 0, 6, 3)

overtime = st.sidebar.selectbox("OverTime", ["No", "Yes"])
business_travel = st.sidebar.selectbox("Business Travel", 
                  ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)
environment_satisfaction = st.sidebar.slider("Environment Satisfaction (1-4)", 1, 4, 3)
work_life_balance = st.sidebar.slider("Work Life Balance (1-4)", 1, 4, 3)
job_involvement = st.sidebar.slider("Job Involvement (1-4)", 1, 4, 3)
relationship_satisfaction = st.sidebar.slider("Relationship Satisfaction (1-4)", 1, 4, 3)
stock_option = st.sidebar.slider("Stock Option Level (0-3)", 0, 3, 1)
job_level = st.sidebar.slider("Job Level (1-5)", 1, 5, 2)
education = st.sidebar.slider("Education (1-5)", 1, 5, 3)

department = st.sidebar.selectbox("Department", 
             ["Sales", "Research & Development", "Human Resources"])
job_role = st.sidebar.selectbox("Job Role", 
           ["Sales Executive", "Research Scientist", "Laboratory Technician",
            "Manufacturing Director", "Healthcare Representative", "Manager",
            "Sales Representative", "Research Director", "Human Resources"])
marital_status = st.sidebar.selectbox("Marital Status", 
                 ["Single", "Married", "Divorced"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education_field = st.sidebar.selectbox("Education Field", 
                  ["Life Sciences", "Medical", "Marketing", 
                   "Technical Degree", "Human Resources", "Other"])

daily_rate = st.sidebar.slider("Daily Rate", 100, 1500, 800)
hourly_rate = st.sidebar.slider("Hourly Rate", 30, 100, 65)
monthly_rate = st.sidebar.slider("Monthly Rate", 2000, 27000, 14000)
percent_hike = st.sidebar.slider("Percent Salary Hike", 11, 25, 15)
performance_rating = st.sidebar.selectbox("Performance Rating", [3, 4])

# ── Encode inputs ──
travel_map = {"Non-Travel": 0, "Travel_Frequently": 1, "Travel_Rarely": 2}
dept_map = {"Human Resources": 0, "Research & Development": 1, "Sales": 2}
edu_map = {"Human Resources": 0, "Life Sciences": 1, "Marketing": 2, 
           "Medical": 3, "Other": 4, "Technical Degree": 5}
gender_map = {"Female": 0, "Male": 1}
role_map = {"Healthcare Representative": 0, "Human Resources": 1,
            "Laboratory Technician": 2, "Manager": 3,
            "Manufacturing Director": 4, "Research Director": 5,
            "Research Scientist": 6, "Sales Executive": 7,
            "Sales Representative": 8}
marital_map = {"Divorced": 0, "Married": 1, "Single": 2}
overtime_map = {"No": 0, "Yes": 1}

input_data = pd.DataFrame([{
    'Age': age,
    'BusinessTravel': travel_map[business_travel],
    'DailyRate': daily_rate,
    'Department': dept_map[department],
    'DistanceFromHome': distance,
    'Education': education,
    'EducationField': edu_map[education_field],
    'EnvironmentSatisfaction': environment_satisfaction,
    'Gender': gender_map[gender],
    'HourlyRate': hourly_rate,
    'JobInvolvement': job_involvement,
    'JobLevel': job_level,
    'JobRole': role_map[job_role],
    'JobSatisfaction': job_satisfaction,
    'MaritalStatus': marital_map[marital_status],
    'MonthlyIncome': monthly_income,
    'MonthlyRate': monthly_rate,
    'NumCompaniesWorked': num_companies,
    'OverTime': overtime_map[overtime],
    'PercentSalaryHike': percent_hike,
    'PerformanceRating': performance_rating,
    'RelationshipSatisfaction': relationship_satisfaction,
    'StockOptionLevel': stock_option,
    'TotalWorkingYears': total_working,
    'TrainingTimesLastYear': training,
    'WorkLifeBalance': work_life_balance,
    'YearsAtCompany': years_company,
    'YearsInCurrentRole': years_role,
    'YearsSinceLastPromotion': years_promoted,
    'YearsWithCurrManager': years_manager
}])[feature_names]

# ── Predict ──
if st.button("🔍 Predict Attrition Risk", use_container_width=True):
    scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if prob >= 0.6:
            st.error(f"🔴 HIGH RISK — {prob:.1%} chance of quitting")
        elif prob >= 0.35:
            st.warning(f"🟡 MEDIUM RISK — {prob:.1%} chance of quitting")
        else:
            st.success(f"🟢 LOW RISK — {prob:.1%} chance of quitting")

    with col2:
        st.metric("Attrition Probability", f"{prob:.1%}")
        st.metric("Prediction", "Will Quit 😟" if pred == 1 else "Will Stay 😊")

    # ── SHAP Explanation ──
    # ── SHAP Explanation ──
    st.markdown("### 🔍 Why this prediction?")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)
    
    fig, ax = plt.subplots(figsize=(12, 3))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=input_data.iloc[0].values,
            feature_names=feature_names
        ), show=False
    )
    st.pyplot(fig)
    plt.close()