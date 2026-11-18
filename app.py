

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()
features = list(model.feature_names_in_)

st.title("🩺 Diabetes Prediction System")
st.write("Random Forest Machine Learning Project")
st.warning(
    "Educational use only. This is not a medical diagnosis."
)

with st.form("patient_form"):
    st.subheader("Patient Health Details")
    values = {}

    col1, col2 = st.columns(2)

    for i, feature in enumerate(features):
        with (col1 if i % 2 == 0 else col2):
            if feature == "BMI":
                values[feature] = st.number_input(
                    "BMI", 10.0, 100.0, 25.0
                )
            elif feature in ["MentHlth", "PhysHlth"]:
                values[feature] = st.number_input(
                    feature, 0, 30, 0
                )
            elif feature == "GenHlth":
                values[feature] = st.selectbox(
                    "General Health (1=Excellent, 5=Poor)",
                    [1, 2, 3, 4, 5]
                )
            elif feature == "Age":
                values[feature] = st.selectbox(
                    "Age Category", list(range(1, 14))
                )
            elif feature == "Education":
                values[feature] = st.selectbox(
                    "Education Category", list(range(1, 7))
                )
            elif feature == "Income":
                values[feature] = st.selectbox(
                    "Income Category", list(range(1, 9))
                )
            else:
                values[feature] = st.selectbox(
                    feature, [0, 1]
                )

    submitted = st.form_submit_button("Predict Diabetes")

if submitted:
    input_df = pd.DataFrame(
        [[values[f] for f in features]],
        columns=features
    )

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Predicted class: Diabetes (1)")
    else:
        st.success("Predicted class: No Diabetes (0)")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        if 1 in classes:
            st.metric(
                "Estimated probability for class 1",
                f"{probs[classes.index(1)]:.1%}"
            )
