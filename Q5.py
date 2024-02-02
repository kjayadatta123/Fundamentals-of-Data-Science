import numpy as np


student_scores = np.array([
    [85, 90, 78, 88],
    [92, 88, 95, 80],
    [78, 85, 90, 88],
    [90, 92, 87, 95]
])

average_scores = np.mean(student_scores, axis=0)
print("Average Scores for Each Subject:", average_scores)

subject_with_highest_average = np.argmax(average_scores)
print("Subject with the Highest Average Score:", subject_with_highest_average)
