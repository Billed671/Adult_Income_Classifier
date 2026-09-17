import os
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000/predict")

st.set_page_config(page_title="Adult Income Predictor", page_icon="💼")
st.title("💼 Adult Income Predictor")
st.write("Fill in the fields and click **Predict** — you’ll find out whether the person earns >50K.")

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 17, 90, 35)
        workclass = st.selectbox("Workclass", [
            "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
            "Local-gov", "State-gov", "Without-pay", "Never-worked"])
        education = st.selectbox("Education", [
            "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
            "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
            "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"])
        education_num = st.number_input("Education num", 1, 16, 13)
        marital_status = st.selectbox("Marital status", [
            "Never-married", "Married-civ-spouse", "Divorced", "Separated",
            "Married-spouse-absent", "Married-AF-spouse", "Widowed"])
        occupation = st.selectbox("Occupation", [
            "Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical",
            "Sales", "Other-service", "Machine-op-inspct", "Transport-moving",
            "Handlers-cleaners", "Farming-fishing", "Tech-support",
            "Protective-serv", "Priv-house-serv", "Armed-Forces"])
        relationship = st.selectbox("Relationship", [
            "Not-in-family", "Husband", "Wife", "Own-child",
            "Unmarried", "Other-relative"])

    with col2:
        race = st.selectbox("Race", [
            "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
        sex = st.selectbox("Sex", ["Male", "Female"])
        capital_gain = st.number_input("Capital gain", 0, 100000, 0)
        capital_loss = st.number_input("Capital loss", 0, 10000, 0)
        hours_per_week = st.number_input("Hours per week", 1, 99, 40)
        native_country = st.selectbox("Native country", [
            "United-States", "Mexico", "Philippines", "Germany", "Canada",
            "India", "England", "Cuba", "Other"])
        fnlwgt = st.number_input("fnlwgt", 10000, 1000000, 100000)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "age": int(age),
        "workclass": workclass,
        "fnlwgt": int(fnlwgt),
        "education": education,
        "education-num": int(education_num),
        "marital-status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital-gain": int(capital_gain),
        "capital-loss": int(capital_loss),
        "hours-per-week": int(hours_per_week),
        "native-country": native_country,
    }
    try:
        r = requests.post(API_URL, json=payload, timeout=15)
        r.raise_for_status()
        res = r.json()
        st.success(f"Prediction: **{res['label']}**  (probability = {res['probability']:.3f})")
    except Exception as e:
        st.error(f"Error calling API: {e}")
