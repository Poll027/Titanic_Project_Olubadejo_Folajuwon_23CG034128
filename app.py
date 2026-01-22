import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Page Config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    .result-card {
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Application Title
st.title("🚢 Titanic Survival Prediction System")
st.markdown("Developed by **Olubadejo Folajuwon (23CG034128)**")
st.write("---")

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "titanic_survival_model.pkl")
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error("Model file not found! Please ensure 'titanic_survival_model.pkl' is in the 'model' directory.")
        return None

model = load_model()

# Sidebar for inputs
with st.sidebar:
    st.header("Passenger Details")
    
    # Input: Pclass
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], format_func=lambda x: f"Class {x}")
    
    # Input: Sex
    sex = st.radio("Sex", ["Male", "Female"])
    
    # Input: Age
    age = st.slider("Age", 0, 100, 30)
    
    # Input: Fare
    fare = st.number_input("Fare Amount ($)", min_value=0.0, max_value=600.0, value=32.0)
    
    # Input: Embarked
    embarked = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

# Validation and Preprocessing
if st.button("Predict Survival"):
    if model:
        # Preprocess inputs
        # Sex: Male=1, Female=0 (Matches common labeling, user must verify training encoding!)
        # UPDATE: Based on my notebook plan: Male=1, Female=0 is typical label encoding alphabetical if F/M? No, F comes before M.
        # Let's align with the notebook code plan:
        # Notebook: le_sex.fit_transform(['female', 'male']) -> female=0, male=1
        sex_encoded = 1 if sex == "Male" else 0
        
        # Embarked: C, Q, S -> Alphabetical order for LabelEncoder commonly
        # C=0, Q=1, S=2
        embarked_map = {"Cherbourg (C)": 0, "Queenstown (Q)": 1, "Southampton (S)": 2}
        embarked_encoded = embarked_map[embarked]
        
        # Create input array
        # Features used in training: ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
        input_data = np.array([[pclass, sex_encoded, age, fare, embarked_encoded]])
        
        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][prediction]
        
        # Display Result
        st.write("### Prediction Result")
        if prediction == 1:
            st.success(f"**Survived** (Confidence: {probability:.2%})")
            st.balloons()
        else:
            st.error(f"**Did Not Survive** (Confidence: {probability:.2%})")
    else:
        st.warning("Model could not be loaded. Please check the files.")

st.write("---")
st.info("Note: This model is for educational purposes.")
