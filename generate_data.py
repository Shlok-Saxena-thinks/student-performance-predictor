import pandas as pd
import random

data = []

for _ in range(500):
    study_hours = random.uniform(0, 10)
    attendance = random.uniform(50, 100)
    previous_marks = random.uniform(30, 95)
    sleep_hours = random.uniform(4, 10)
    gaming_hours = random.uniform(0, 5)
    sports_hours = random.uniform(0, 3)

    final_marks = (
        study_hours * 5 +
        attendance * 0.2 +
        previous_marks * 0.5 +
        (8 - abs(sleep_hours - 7)) * 2 -   # best sleep around 7
        gaming_hours * 2 +
        sports_hours * 1.5 +
        random.uniform(-5, 5)  # randomness
    )

    final_marks = max(0, min(100, final_marks))

    data.append([
        study_hours,
        attendance,
        previous_marks,
        sleep_hours,
        gaming_hours,
        sports_hours,
        final_marks
    ])

df = pd.DataFrame(data, columns=[
    "StudyHours",
    "Attendance",
    "PreviousMarks",
    "SleepHours",
    "GamingHours",
    "SportsHours",
    "FinalMarks"
])

df.to_csv("student_data.csv", index=False)

print("DataSet Compeleted")
