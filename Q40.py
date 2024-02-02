import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a sample dataset
data = {
    'Name': ['Messi', 'Ronaldo', 'Neymar', 'Lewandowski', 'De Bruyne', 'Modric', 'Van Dijk', 'Salah', 'Ramos', 'Mbappe'],
    'Age': [34, 36, 29, 33, 30, 35, 30, 29, 35, 23],
    'Position': ['Forward', 'Forward', 'Forward', 'Forward', 'Midfielder', 'Midfielder', 'Defender', 'Forward', 'Defender', 'Forward'],
    'GoalsScored': [30, 28, 20, 35, 12, 8, 5, 25, 6, 22],
    'WeeklySalary': [750000, 800000, 600000, 550000, 350000, 300000, 250000, 700000, 400000, 650000]
}

df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('soccer_players.csv', index=False)

# Read the dataset from the CSV file
df = pd.read_csv('soccer_players.csv')

# Find the top 5 players with the highest number of goals scored
top_goals_players = df.nlargest(5, 'GoalsScored')

# Find the top 5 players with the highest salaries
top_salary_players = df.nlargest(5, 'WeeklySalary')

# Calculate the average age of players
average_age = df['Age'].mean()

# Display the names of players who are above the average age
above_average_age_players = df[df['Age'] > average_age]['Name']

# Visualize the distribution of players based on their positions using a bar chart
position_distribution = df['Position'].value_counts()
position_distribution.plot(kind='bar', color='skyblue')
plt.title('Distribution of Players Based on Positions')
plt.xlabel('Position')
plt.ylabel('Number of Players')
plt.show()

# Display the results
print("Top 5 Players with the Highest Number of Goals Scored:")
print(top_goals_players[['Name', 'GoalsScored']])
print("\nTop 5 Players with the Highest Salaries:")
print(top_salary_players[['Name', 'WeeklySalary']])
print(f"\nAverage Age of Players: {average_age:.2f}")
print("\nPlayers Above the Average Age:")
print(above_average_age_players)
