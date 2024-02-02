import pandas as pd

data = {
    'City': ['Mumbai', 'Delhi', 'Bangalore'],
    'Jan_Temp': [22, '15a', 18],
    'Feb_Temp': [25, 18, 20],
    'Mar_Temp': [28, 20, 22],
}

df = pd.DataFrame(data)

# Convert non-numeric values to NaN only for temperature columns
df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

# Calculate mean temperature for each city, skipping non-numeric values
df['Mean_Temperature'] = df.iloc[:, 1:].mean(axis=1, skipna=True)
df['Temp_Std_Dev'] = df.iloc[:, 1:-1].std(axis=1)
df['Temp_Range'] = df.iloc[:, 1:-2].apply(lambda row: row.max() - row.min(), axis=1)

city_with_highest_range = df.loc[df['Temp_Range'].idxmax()]['City']
most_consistent_city = df.loc[df['Temp_Std_Dev'].idxmin()]['City']

print(df[['City', 'Mean_Temperature', 'Temp_Std_Dev', 'Temp_Range']])
print(f"\nThe city with the highest temperature range is: {city_with_highest_range}")
print(f"The city with the most consistent temperature is: {most_consistent_city}")
