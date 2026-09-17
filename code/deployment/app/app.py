import os
import random
import requests
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000/predict")

st.set_page_config(page_title="Adult Income Predictor", page_icon="💼")
st.title("💼 Adult Income Predictor")
st.write("Fill in the fields and click **Predict** — you'll find out whether the person earns >50K.")

# ---- Допустимые значения ----
WORKCLASSES = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
               "Local-gov", "State-gov", "Without-pay", "Never-worked"]

EDUCATIONS = ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
              "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
              "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]

EDUCATION_NUM = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
    "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
    "Prof-school": 15, "Doctorate": 16,
}

MARITAL_STATUSES = ["Never-married", "Married-civ-spouse", "Divorced",
                    "Separated", "Married-spouse-absent", "Married-AF-spouse",
                    "Widowed"]

OCCUPATIONS = ["Prof-specialty", "Craft-repair", "Exec-managerial",
               "Adm-clerical", "Sales", "Other-service", "Machine-op-inspct",
               "Transport-moving", "Handlers-cleaners", "Farming-fishing",
               "Tech-support", "Protective-serv", "Priv-house-serv", "Armed-Forces"]

RELATIONSHIPS = ["Not-in-family", "Husband", "Wife", "Own-child",
                 "Unmarried", "Other-relative"]

RACES = ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"]

SEXES = ["Male", "Female"]

COUNTRIES = ["United-States", "Mexico", "Philippines", "Germany", "Canada",
             "India", "England", "Cuba", "Other"]

# Колонки, которые API ожидает на вход (без target)
REQUIRED_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
]

# Числовые колонки — их надо конвертировать в int перед отправкой
NUMERIC_COLUMNS = [
    "age", "fnlwgt", "education-num",
    "capital-gain", "capital-loss", "hours-per-week",
]

DEFAULTS = {
    "age": 35,
    "workclass": "Private",
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Prof-specialty",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
    "fnlwgt": 100000,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def randomize():
    age = random.choices(
        population=[random.randint(20, 40), random.randint(41, 60), random.randint(61, 75)],
        weights=[5, 4, 1],
    )[0]
    education = random.choice(EDUCATIONS)
    sex = random.choices(SEXES, weights=[2, 1])[0]
    race = random.choices(RACES, weights=[85, 9, 3, 1, 2])[0]
    country = random.choices(COUNTRIES, weights=[90, 3, 1, 1, 1, 1, 1, 1, 1])[0]
    capital_gain = 0 if random.random() < 0.9 else random.choice(
        [random.randint(1000, 9999), random.randint(10000, 50000)])
    capital_loss = 0 if random.random() < 0.95 else random.randint(1000, 4000)

    st.session_state.update({
        "age": age,
        "workclass": random.choice(WORKCLASSES),
        "education": education,
        "education_num": EDUCATION_NUM[education],
        "marital_status": random.choice(MARITAL_STATUSES),
        "occupation": random.choice(OCCUPATIONS),
        "relationship": random.choice(RELATIONSHIPS),
        "race": race,
        "sex": sex,
        "capital_gain": capital_gain,
        "capital_loss": capital_loss,
        "hours_per_week": random.randint(20, 60),
        "native_country": country,
        "fnlwgt": random.randint(50000, 400000),
    })


def call_api(payload):
    """Отправляет один объект на API. Возвращает dict с полями label/probability или None."""
    try:
        r = requests.post(API_URL, json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def row_to_payload(row):
    """Преобразует строку DataFrame в dict, готовый к отправке в API."""
    payload = {}
    for col in REQUIRED_COLUMNS:
        val = row[col]
        if pd.isna(val):
            return None
        if col in NUMERIC_COLUMNS:
            payload[col] = int(val)
        else:
            payload[col] = str(val)
    return payload


# ----------------- Вкладки -----------------
tab_single, tab_batch = st.tabs(["Single prediction", "Batch CSV upload"])

# ---------- Вкладка 1: одна запись ----------
with tab_single:
    col_a, col_b = st.columns([1, 4])
    with col_a:
        st.button("🎲 Randomize", on_click=randomize, use_container_width=True)
    with col_b:
        st.caption("Fills the form with plausible random values.")

    with st.form("input_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.number_input("Age", 17, 90, key="age")
            st.selectbox("Workclass", WORKCLASSES, key="workclass")
            st.selectbox("Education", EDUCATIONS, key="education")
            st.number_input("Education num", 1, 16, key="education_num")
            st.selectbox("Marital status", MARITAL_STATUSES, key="marital_status")
            st.selectbox("Occupation", OCCUPATIONS, key="occupation")
            st.selectbox("Relationship", RELATIONSHIPS, key="relationship")

        with col2:
            st.selectbox("Race", RACES, key="race")
            st.selectbox("Sex", SEXES, key="sex")
            st.number_input("Capital gain", 0, 100000, key="capital_gain")
            st.number_input("Capital loss", 0, 10000, key="capital_loss")
            st.number_input("Hours per week", 1, 99, key="hours_per_week")
            st.selectbox("Native country", COUNTRIES, key="native_country")
            st.number_input("fnlwgt", 10000, 1000000, key="fnlwgt")

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "age": int(st.session_state["age"]),
            "workclass": st.session_state["workclass"],
            "fnlwgt": int(st.session_state["fnlwgt"]),
            "education": st.session_state["education"],
            "education-num": int(st.session_state["education_num"]),
            "marital-status": st.session_state["marital_status"],
            "occupation": st.session_state["occupation"],
            "relationship": st.session_state["relationship"],
            "race": st.session_state["race"],
            "sex": st.session_state["sex"],
            "capital-gain": int(st.session_state["capital_gain"]),
            "capital-loss": int(st.session_state["capital_loss"]),
            "hours-per-week": int(st.session_state["hours_per_week"]),
            "native-country": st.session_state["native_country"],
        }
        res = call_api(payload)
        if "error" in res:
            st.error(f"Error calling API: {res['error']}")
        else:
            st.success(f"Prediction: **{res['label']}**  (probability = {res['probability']:.3f})")


# ---------- Вкладка 2: пакетная загрузка CSV ----------
with tab_batch:
    st.write(
        "Upload a CSV file with test data. The file must contain the following columns:"
    )
    st.code(", ".join(REQUIRED_COLUMNS), language="text")

    st.write(
        "The target column `income` is optional. If it is present, it will be shown "
        "next to the predictions for comparison."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read the file: {e}")
            df = None

        if df is not None:
            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            elif len(df) == 0:
                st.warning("The file contains no rows.")
            else:
                st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns.")
                st.dataframe(df.head(10), use_container_width=True)

                max_rows = st.number_input(
                    "How many rows to predict (start from the top)",
                    min_value=1,
                    max_value=len(df),
                    value=min(len(df), 50),
                    step=1,
                )

                if st.button("Run predictions", type="primary"):
                    df_subset = df.head(int(max_rows)).copy()
                    results = []
                    errors = 0

                    progress = st.progress(0.0, text="Calling API...")
                    for i, (_, row) in enumerate(df_subset.iterrows()):
                        payload = row_to_payload(row)
                        if payload is None:
                            results.append({**row.to_dict(),
                                            "prediction": "skipped (NaN)",
                                            "probability": None})
                            errors += 1
                        else:
                            res = call_api(payload)
                            if "error" in res:
                                results.append({**row.to_dict(),
                                                "prediction": "error",
                                                "probability": None})
                                errors += 1
                            else:
                                results.append({**row.to_dict(),
                                                "prediction": res["label"],
                                                "probability": res["probability"]})
                        progress.progress((i + 1) / len(df_subset),
                                          text=f"Row {i + 1}/{len(df_subset)}")
                    progress.empty()

                    result_df = pd.DataFrame(results)

                    # Короткая сводка
                    ok = result_df[result_df["prediction"].isin(["<=50K", ">50K"])]
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Rows predicted", len(ok))
                    col2.metric("Errors / skipped", errors)
                    if len(ok) > 0:
                        share = (ok["prediction"] == ">50K").mean()
                        col3.metric("Share >50K", f"{share:.1%}")

                    st.write("Results")
                    st.dataframe(result_df, use_container_width=True)

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download results as CSV",
                        data=csv_bytes,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )
