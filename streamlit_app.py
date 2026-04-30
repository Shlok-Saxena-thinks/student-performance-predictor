import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()
# Load data
df = pd.read_csv("student_data.csv")

X = df[[
    "StudyHours",
    "Attendance",
    "PreviousMarks",
    "SleepHours",
    "GamingHours",
    "SportsHours"
]]
y = df["FinalMarks"]


st.title("🎓 Student Performance Predictor")

# Inputs
study_hours = st.slider("Study Hours", 0.0, 10.0, 5.0)
attendance = st.slider("Attendance (%)", 0.0, 100.0, 75.0)
previous_marks = st.slider("Previous Marks", 0.0, 100.0, 60.0)
sleep_hours = st.slider("Sleep Hours", 0.0, 10.0, 7.0)
gaming_hours = st.slider("Gaming Hours", 0.0, 5.0, 1.0)
sports_hours = st.slider("Sports Hours", 0.0, 3.0, 1.0)

# Prediction
if st.button("Predict"):
    sample = pd.DataFrame([{
        "StudyHours": study_hours,
        "Attendance": attendance,
        "PreviousMarks": previous_marks,
        "SleepHours": sleep_hours,
        "GamingHours": gaming_hours,
        "SportsHours": sports_hours
    }])

    prediction = model.predict(sample)[0]
    prediction = max(0, min(100, prediction))

    st.success(f"Predicted Marks: {prediction:.2f}")

    # Interpretation
    if prediction < 50:
        st.error("Performance is Low")
    elif prediction < 75:
        st.warning("Average Performance")
    else:
        st.success("Excellent Performance")

    # Input Summary
    st.write("### Input Summary")
    st.write(sample)

# Feature Importance (always visible)
st.write("### Feature Importance")

importance = model.coef_
features = X.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

st.bar_chart(importance_df.set_index("Feature"))