import streamlit as st
import pandas as pd
import joblib

model=joblib.load("emi_regressor.joblib")

st.title("Maximum EMI Amount Prediction")

# Input widgets for all features
age = st.number_input("Age", 18, 100, 30)
gender = st.selectbox("Gender (0=Female, 1=Male)", [0, 1], index=1)
marital_status = st.number_input("Marital Status (encoded)", 0, 10, 0)
education = st.number_input("Education (encoded)", 0, 10, 0)
monthly_salary = st.number_input("Monthly Salary", 0, 1_000_000, 50000)
employment_type = st.number_input("Employment Type (encoded)", 0, 10, 0)
years_of_employment = st.number_input("Years of Employment", 0, 50, 5)
company_type = st.number_input("Company Type (encoded)", 0, 10, 0)
house_type = st.number_input("House Type (encoded)", 0, 10, 0)
monthly_rent = st.number_input("Monthly Rent", 0, 1_000_000, 1000)
family_size = st.number_input("Family Size", 1, 20, 3)
dependents = st.number_input("Number of Dependents", 0, 10, 0)
school_fees = st.number_input("School Fees", 0, 1_000_000, 0)
college_fees = st.number_input("College Fees", 0, 1_000_000, 0)
travel_expenses = st.number_input("Travel Expenses", 0, 1_000_000, 0)
groceries_utilities = st.number_input("Groceries and Utilities", 0, 1_000_000, 0)
other_monthly_expenses = st.number_input("Other Monthly Expenses", 0, 1_000_000, 0)
existing_loans = st.number_input("Existing Loans (encoded)", 0, 10, 0)
current_emi_amount = st.number_input("Current EMI Amount", 0, 1_000_000, 0)
credit_score = st.number_input("Credit Score", 0, 1000, 700)
bank_balance = st.number_input("Bank Balance", 0, 1_000_000, 10000)
emergency_fund = st.number_input("Emergency Fund", 0, 1_000_000, 0)
emi_scenario = st.number_input("EMI Scenario (encoded)", 0, 10, 0)
requested_amount = st.number_input("Requested Amount", 0, 1_000_000, 100000)
requested_tenure = st.number_input("Requested Tenure (months)", 1, 360, 60)
debt_to_income_ratio = st.number_input("Debt to Income Ratio", 0.0, 10.0, 0.3, format="%.2f")
total_expenses = st.number_input("Total Expenses", 0, 1_000_000, 0)
expense_to_income_ratio = st.number_input("Expense to Income Ratio", 0.0, 10.0, 0.5, format="%.2f")
affordability_ratio = st.number_input("Affordability Ratio", 0.0, 10.0, 0.5, format="%.2f")
emergency_fund_ratio = st.number_input("Emergency Fund Ratio", 0.0, 10.0, 0.0, format="%.2f")
credit_score_band = st.number_input("Credit Score Band (encoded)", 0, 10, 0)
employment_stability = st.number_input("Employment Stability (encoded)", 0, 10, 0)
monthly_salary_log = st.number_input("Monthly Salary (Log Transformed)", 0.0, 15.0, 10.0, format="%.4f")

if st.button("Predict Max EMI"):
    input_dict = {
        'age': age,
        'gender':gender,
        'gender_encoded': gender,
        'marital_status': marital_status,
        'education': education,
        'monthly_salary': monthly_salary,
        'employment_type': employment_type,
        'years_of_employment': years_of_employment,
        'company_type': company_type,
        'house_type': house_type,
        'monthly_rent': monthly_rent,
        'family_size': family_size,
        'dependents': dependents,
        'school_fees': school_fees,
        'college_fees': college_fees,
        'travel_expenses': travel_expenses,
        'groceries_utilities': groceries_utilities,
        'other_monthly_expenses': other_monthly_expenses,
        'existing_loans': existing_loans,
        'current_emi_amount': current_emi_amount,
        'credit_score': credit_score,
        'bank_balance': bank_balance,
        'emergency_fund': emergency_fund,
        'emi_scenario': emi_scenario,
        'requested_amount': requested_amount,
        'requested_tenure': requested_tenure,
        'debt_to_income_ratio': debt_to_income_ratio,
        'total_expenses': total_expenses,
        'expense_to_income_ratio': expense_to_income_ratio,
        'affordability_ratio': affordability_ratio,
        'emergency_fund_ratio': emergency_fund_ratio,
        'credit_score_band': credit_score_band,
        'employment_stability': employment_stability,
        'monthly_salary_log': monthly_salary_log
    }
    input_df = pd.DataFrame([input_dict])

    float_cols = [
        'age', 'monthly_salary', 'years_of_employment', 'monthly_rent', 'school_fees',
        'college_fees', 'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
        'current_emi_amount', 'credit_score', 'bank_balance', 'emergency_fund', 'requested_amount',
        'debt_to_income_ratio', 'total_expenses', 'expense_to_income_ratio', 'affordability_ratio',
        'emergency_fund_ratio', 'monthly_salary_log'
    ]

    input_df[float_cols] = input_df[float_cols].astype('float64')

    int_cols = [
        'gender', 'marital_status', 'education', 'employment_type', 'company_type', 'house_type',
        'family_size', 'dependents', 'existing_loans', 'emi_scenario', 'requested_tenure',
        'credit_score_band', 'employment_stability', 'gender_encoded'
    ]

    input_df[int_cols] = input_df[int_cols].astype('int64')

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Maximum EMI Amount: {prediction:.2f}")
