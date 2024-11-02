### Data Cleaning & Preprocessing
## 1.Load a dataset with mixed data types and clean the data by removing or correcting any non-numeric values
import pandas as pd

# Sample DataFrame with mixed data types
data = {
    'Column1': [1, 2, 'three', 4, 5],
    'Column2': [1.1, 'two', 3.3, 'four', 5.5]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Function to convert values to numeric, replacing non-numeric values with NaN
def convert_to_numeric(value):
    try:
        return pd.to_numeric(value)
    except ValueError:
        return None  # or use np.nan if you prefer

# Apply the conversion function to the DataFrame
df_cleaned = df.applymap(convert_to_numeric)
print("\nCleaned DataFrame:")
print(df_cleaned)

## 2.Identify and remove duplicate rows from a DataFrame
import pandas as pd

# Sample DataFrame with duplicate rows
data = {
    'Column1': [1, 2, 2, 3, 4, 4],
    'Column2': ['A', 'B', 'B', 'C', 'D', 'D']
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Identify and remove duplicate rows
df_cleaned = df.drop_duplicates()
print("\nDataFrame after removing duplicates:")
print(df_cleaned)

## 3.Normalize a dataset using StandardScaler from scikit-learn and explain the difference between min-max scaling and standard scaling
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Sample DataFrame
data = {'Column1': [1, 2, 3, 4, 5], 'Column2': [10, 20, 30, 40, 50]}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Initialize StandardScaler
scaler = StandardScaler()

# Fit and transform the data
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
print("\nStandard Scaled DataFrame:")
print(scaled_df)

## 4.Convert a categorical column to numeric using label encoding and explain its impact on the dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Sample DataFrame with a categorical column
data = {
    'Category': ['A', 'B', 'A', 'C', 'B', 'C', 'A']
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the categorical column
df['Category'] = label_encoder.fit_transform(df['Category'])
print("\nDataFrame after label encoding:")
print(df)

## 5.Split a dataset into training and testing sets using an 80-20 split and shuffle the data before splitting
import pandas as pd
from sklearn.model_selection import train_test_split

# Sample DataFrame
data = {
    'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'Label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Define features and labels
X = df[['Feature1', 'Feature2']]
y = df['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

print("\nTraining features:")
print(X_train)
print("\nTesting features:")
print(X_test)
print("\nTraining labels:")
print(y_train)
print("\nTesting labels:")
print(y_test)
