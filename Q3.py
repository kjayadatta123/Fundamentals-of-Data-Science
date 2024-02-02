import pandas as pd

data = {
    'Student_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Course': ['Math', 'Math', 'Math', 'Science', 'Science', 'Science', 'History', 'History', 'History', 'History'],
    'Score': [85, 90, 75, 78, 85, 80, 92, 88, 75, 80],
    'Hours_Studied': [5, 7, 3, 4, 6, 5, 8, 7, 4, 6]
}
student_data = pd.DataFrame(data)
correlation_per_course = student_data.groupby('Course')[['Hours_Studied', 'Score']].corr().iloc[0::2,-1].reset_index()
correlation_per_course.rename(columns={'Score': 'Correlation'}, inplace=True)

print("Correlation between Hours Studied and Score for each course:")
print(correlation_per_course)
print("\n")
strongest_corr_course = correlation_per_course.loc[correlation_per_course.groupby('Course')['Correlation'].idxmax()]
weakest_corr_course = correlation_per_course.loc[correlation_per_course.groupby('Course')['Correlation'].idxmin()]

print("Course with strongest correlation:")
print(strongest_corr_course)
print("\nCourse with weakest correlation:")
print(weakest_corr_course)
print("\n")
average_scores_hours = student_data.groupby('Course').agg({'Score': 'mean', 'Hours_Studied': 'mean'}).reset_index()
average_scores_hours.rename(columns={'Score': 'Average_Score', 'Hours_Studied': 'Average_Hours_Studied'}, inplace=True)

print("Average score and hours studied for each course:")
print(average_scores_hours)
