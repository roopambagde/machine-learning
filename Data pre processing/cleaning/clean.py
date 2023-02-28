import pandas as pd

# Load the data into a pandas dataframe
df = pd.read_csv("data.csv")

# Remove duplicates
df = df.drop_duplicates()

# Fill missing values with 0
df = df.fillna(0)

# Replace specific values
df['Salary'].replace(['old_value_1', 'old_value_2'], ['new_value_1', 'new_value_2'], inplace=True)

# Remove outliers using Z-score
mean = df['Salary'].mean()
std = df['Salary'].std()
z = (df['Salary'] - mean) / std
df = df[(z < 3) & (z > -3)]

# Save the cleaned data
df.to_csv("cleaned_data.csv", index=False)
