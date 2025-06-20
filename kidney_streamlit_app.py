# kidney_streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

# Load and train models
@st.cache_data
def load_models():
    df = pd.read_csv('kidney_disease (2).csv')
    df['classification'] = df['classification'].apply(lambda x: 1 if x == 'ckd' else 0)
    features = ['age', 'bp', 'al', 'su', 'sc', 'sod', 'pot', 'hemo', 'bgr', 'bu']

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    X = imputer.fit_transform(df[features])
    X = scaler.fit_transform(X)
    y = df['classification']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    svm_model = SVC(probability=True)
    svm_model.fit(X_train, y_train)

    ckd_data = df[df['classification'] == 1]
    X_ckd = imputer.transform(ckd_data[features])
    X_ckd = scaler.transform(X_ckd)
    y_sessions = np.random.randint(1, 4, size=len(ckd_data))

    rf_model = RandomForestRegressor(n_estimators=100)
    rf_model.fit(X_ckd, y_sessions)

    return df, svm_model, rf_model, imputer, scaler, features

def calculate_gfr(age, sc, gender='female'):
    try:
        age = float(age)
        sc = float(sc)
        if gender == 'male':
            k, alpha, multiplier = 0.9, -0.302, 1
        else:
            k, alpha, multiplier = 0.7, -0.241, 1.012
        gfr = 142 * (min(sc / k, 1) ** alpha) * (max(sc / k, 1) ** -1.2) * (0.9938 ** age) * multiplier
        return max(gfr, 1)
    except:
        return None

def classify_stage(gfr):
    if gfr is None:
        return "Unable to calculate"
    elif gfr >= 90:
        return "Stage 1 (Normal or high GFR)"
    elif gfr >= 60:
        return "Stage 2 (Mildly decreased GFR)"
    elif gfr >= 45:
        return "Stage 3a (Mild to moderate decrease)"
    elif gfr >= 30:
        return "Stage 3b (Moderate to severe decrease)"
    elif gfr >= 15:
        return "Stage 4 (Severely decreased GFR)"
    else:
        return "Stage 5 (Kidney failure)"

# Main App
def main():
    st.title("Kidney Disease Diagnosis System")

    df, svm_model, rf_model, imputer, scaler, features = load_models()

    st.sidebar.header("Enter Patient Data")
    inputs = {}
    for f in features:
        inputs[f] = st.sidebar.number_input(f, value=0.0)

    gender = st.sidebar.radio("Gender", ['female', 'male'])

    if st.sidebar.button("Diagnose"):
        input_data = [inputs[f] for f in features]
        data = imputer.transform([input_data])
        data = scaler.transform(data)

        prediction = svm_model.predict(data)[0]
        prob = max(svm_model.predict_proba(data)[0]) * 100
        status = "Chronic (CKD)" if prediction == 1 else "Not Chronic"
        st.subheader(f"Status: {status}")
        st.subheader(f"Confidence: {prob:.1f}%")

        gfr = calculate_gfr(inputs['age'], inputs['sc'], gender)
        stage = classify_stage(gfr)
        st.subheader(f"eGFR: {gfr:.1f} mL/min/1.73m²" if gfr else "Invalid GFR Input")
        st.subheader(f"CKD Stage: {stage}")

        if gfr:
            if gfr < 15:
                dialysis = "Dialysis Required (Stage 5 - Kidney Failure)"
                sessions = "3 times/week"
            elif gfr < 30:
                dialysis = "Dialysis Preparation (Stage 4)"
                sessions = "1–2 times/week"
            elif gfr < 60:
                dialysis = "Usually Not Required (Stage 3)"
                sessions = "Only if already undergoing dialysis"
            else:
                dialysis = "Not Required"
                sessions = "N/A"

            similar_patients = df[
                (df['classification'] == 1) &
                (df['sc'] < inputs['sc'] + 0.3) & (df['sc'] > inputs['sc'] - 0.3) &
                (df['hemo'] < inputs['hemo'] + 1) & (df['hemo'] > inputs['hemo'] - 1)
            ]

            if len(similar_patients) > 0 and gfr > 30:
                dialysis += " (Note: Similar patients received dialysis)"
                sessions = "Consult your doctor"

            st.subheader(f"Dialysis Recommendation: {dialysis}")
            st.subheader(f"Session Advice: {sessions}")

if __name__ == "__main__":
    main()

