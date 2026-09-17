import os
import random
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://api:8000/predict")

st.set_page_config(page_title="Adult Income Predictor", page_icon="💼")
st.title("💼 Adult Income Predictor")
st.write("Fill in the fields and click **Predict** — you'll find out whether the person earns >50K.")

# ---- Допустимые значения (используются и в полях, и в рандомайзере) ----
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

# Значения по умолчанию
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

# Инициализация session_state при первом запуске
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def randomize():
    """Заполняет форму правдоподобными случайными значениями."""
    # Возраст — обычно 20–60, но иногда до 70
    age = random.choices(
        population=[random.randint(20, 40), random.randint(41, 60), random.randint(61, 75)],
        weights=[5, 4, 1],
    )[0]

    # Образование выбирается случайно, education_num согласован
    education = random.choice(EDUCATIONS)

    # Пол: ~2/3 мужчин (как в датасете)
    sex = random.choices(SEXES, weights=[2, 1])[0]

    # Раса: преимущественно White (реалистичное распределение)
    race = random.choices(
        RACES,
        weights=[85, 9, 3, 1, 2],
    )[0]

    # Страна: преимущественно United-States
    country = random.choices(
        COUNTRIES,
        weights=[90, 3, 1, 1, 1, 1, 1, 1, 1],
    )[0]

    # Capital gain/loss: у большинства 0
    if random.random() < 0.9:
        capital_gain = 0
    else:
        capital_gain = random.choice([random.randint(1000, 9999),
                                      random.randint(10000, 50000)])

    if random.random() < 0.95:
        capital_loss = 0
    else:
        capital_loss = random.randint(1000, 4000)

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


# Кнопка Randomize — вне формы, чтобы могла менять поля
col_a, col_b = st.columns([1, 4])
with col_a:
    st.button("🎲 Randomize", on_click=randomize, use_container_width=True)
with col_b:
    st.caption("Fills the form with plausible random values. Useful for testing the model.")


with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", 17, 90, key="age")
        workclass = st.selectbox("Workclass", WORKCLASSES, key="workclass")
        education = st.selectbox("Education", EDUCATIONS, key="education")
        education_num = st.number_input("Education num", 1, 16, key="education_num")
        marital_status = st.selectbox("Marital status", MARITAL_STATUSES, key="marital_status")
        occupation = st.selectbox("Occupation", OCCUPATIONS, key="occupation")
        relationship = st.selectbox("Relationship", RELATIONSHIPS, key="relationship")

    with col2:
        race = st.selectbox("Race", RACES, key="race")
        sex = st.selectbox("Sex", SEXES, key="sex")
        capital_gain = st.number_input("Capital gain", 0, 100000, key="capital_gain")
        capital_loss = st.number_input("Capital loss", 0, 10000, key="capital_loss")
        hours_per_week = st.number_input("Hours per week", 1, 99, key="hours_per_week")
        native_country = st.selectbox("Native country", COUNTRIES, key="native_country")
        fnlwgt = st.number_input("fnlwgt", 10000, 1000000, key="fnlwgt")

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
    try:
        r = requests.post(API_URL, json=payload, timeout=15)
        r.raise_for_status()
        res = r.json()
        st.success(f"Prediction: **{res['label']}**  (probability = {res['probability']:.3f})")
    except Exception as e:
        st.error(f"Error calling API: {e}")
