import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
df = pd.read_csv("student_data.csv")

# Features and target
X = df[[
    "StudyHours",
    "Attendance",
    "PreviousMarks",
    "SleepHours",
    "GamingHours",
    "SportsHours"
]]
y = df["FinalMarks"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy
score = model.score(X_test, y_test)
print("Model Accuracy (R² score):", score)

sample = pd.DataFrame([{
    "StudyHours": 7,
    "Attendance": 90,
    "PreviousMarks": 80,
    "SleepHours": 7,
    "GamingHours": 1,
    "SportsHours": 1
}])

prediction = model.predict(sample)

predicted_marks = max(0, min(100, prediction[0]))
print("Predicted Marks:", predicted_marks)
# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as model.pkl")