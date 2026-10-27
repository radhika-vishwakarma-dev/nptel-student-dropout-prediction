import pandas as pd
import numpy as np

# Set seed for reproducible synthetic data
np.random.seed(42)

# Set total dataset size to 10,000 records across 5 years
n_students = 10000

years = np.random.choice([2021, 2022, 2023, 2024, 2025], size=n_students)
courses = np.random.choice(
    ['Data Structures', 'Machine Learning', 'Cloud Computing', 'Database Systems', 'Python Programming'],
    size=n_students
)

# Core metrics specified in project document
quiz_scores = np.random.uniform(20, 100, size=n_students)
assignment_scores = np.random.uniform(30, 100, size=n_students)
video_watch_time = np.random.uniform(10, 100, size=n_students)
login_frequency = np.random.randint(1, 15, size=n_students)
forum_participation = np.random.randint(0, 20, size=n_students)
attendance = np.random.uniform(30, 100, size=n_students)

# Realistic risk score calculation driving dropout logic
dropout_score = (
    (100 - quiz_scores) * 0.25 +
    (100 - video_watch_time) * 0.25 +
    (100 - attendance) * 0.20 +
    (15 - login_frequency) * 2 +
    (100 - assignment_scores) * 0.15
)

# Target Status: Completed vs Dropped
final_status = np.where(dropout_score > 38, 'Dropped', 'Completed')

# Build DataFrame strictly matching required schema
data = pd.DataFrame({
    'Student_ID': [f'NPTEL_{10000 + i}' for i in range(n_students)],
    'Year': years,
    'Course': courses,
    'Assignment_Score': np.round(assignment_scores, 2),
    'Quiz_Score': np.round(quiz_scores, 2),
    'Video_Watch_Time': np.round(video_watch_time, 2),
    'Login_Frequency': login_frequency,
    'Forum_Participation': forum_participation,
    'Attendance': np.round(attendance, 2),
    'Final_Status': final_status
})

# Save to CSV
data.to_csv('nptel_student_data.csv', index=False)
print("SUCCESS: 5-year 'nptel_student_data.csv' (10,000 records) has been generated!")