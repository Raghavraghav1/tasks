### Pandas
## 1.Load a CSV file into a Pandas DataFrame and display the first 10 rowsimport pandas as pd

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('your_file.csv')

# Display the first 10 rows
print(df.head(10))

## 2.Perform groupby operations on a dataset to find the mean and sum of numerical columns based on a categorical column
import pandas as pd

# Sample dataset
data = {
    'Category': ['A', 'B', 'A', 'B', 'A', 'B'],
    'Value1': [10, 20, 30, 40, 50, 60],
    'Value2': [1, 2, 3, 4, 5, 6]
}

# Create a DataFrame
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Group by 'Category' and calculate the mean and sum
grouped = df.groupby('Category').agg({'Value1': ['mean', 'sum'], 'Value2': ['mean', 'sum']})
print("\nGrouped DataFrame with mean and sum:")
print(grouped)

## 3.Handle missing data in a DataFrame by replacing NaN values with the column mean
import pandas as pd
import numpy as np

# Sample DataFrame with missing values
data = {
    'Column1': [1, 2, np.nan, 4, 5],
    'Column2': [np.nan, 2, 3, np.nan, 5]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Replace NaN values with the column mean
df_filled = df.apply(lambda col: col.fillna(col.mean()))
print("\nDataFrame after replacing NaN values with column mean:")
print(df_filled)

## 4.Merge two DataFrames on a common key and perform an inner, outer, left, and right join
import pandas as pd

# Sample DataFrames
data1 = {
    'Key': ['A', 'B', 'C'],
    'Value1': [1, 2, 3]
}

data2 = {
    'Key': ['B', 'C', 'D'],
    'Value2': [4, 5, 6]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# Perform an inner join
inner_join = pd.merge(df1, df2, on='Key', how='inner')
print("\nInner Join:")
print(inner_join)

# Perform an outer join
outer_join = pd.merge(df1, df2, on='Key', how='outer')
print("\nOuter Join:")
print(outer_join)

# Perform a left join
left_join = pd.merge(df1, df2, on='Key', how='left')
print("\nLeft Join:")
print(left_join)

# Perform a right join
right_join = pd.merge(df1, df2, on='Key', how='right')
print("\nRight Join:")
print(right_join)

## 5.Convert a column of object type into a float type and handle any errors that occur during conversion
import pandas as pd

# Sample DataFrame with an object column
data = {'Column1': ['1.1', '2.2', 'three', '4.4', '5.5']}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Define a function to convert to float and handle errors
def safe_convert_to_float(value):
    try:
        return float(value)
    except ValueError:
        return None  # or you can use np.nan if you prefer

# Apply the conversion function to the column
df['Column1'] = df['Column1'].apply(safe_convert_to_float)
print("\nDataFrame after conversion:")
print(df)

## 6.Filter a DataFrame to select rows where a specific column's values fall within a given range
import pandas as pd

# Sample DataFrame
data = {
    'Column1': [1, 5, 7, 10, 15, 20, 25],
    'Column2': [2, 6, 8, 11, 16, 21, 26]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Define the range
lower_bound = 5
upper_bound = 20

# Filter the DataFrame
filtered_df = df[(df['Column1'] >= lower_bound) & (df['Column1'] <= upper_bound)]
print("\nFiltered DataFrame:")
print(filtered_df)

## 7.Create a pivot table from a DataFrame and analyze the data based on multiple aggregations
import pandas as pd

# Sample DataFrame
data = {
    'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Subcategory': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
    'Values': [10, 15, 10, 20, 30, 25]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Create a pivot table
pivot_table = pd.pivot_table(df, 
                             values='Values', 
                             index=['Category'], 
                             columns=['Subcategory'], 
                             aggfunc={'Values': ['sum', 'mean']})
print("\nPivot Table with Multiple Aggregations:")
print(pivot_table)

## 8.Use the apply() function to apply a custom function to each element in a Pandas Series
import pandas as pd

# Sample Series
data = pd.Series([1, 2, 3, 4, 5])

# Define a custom function
def square(x):
    return x ** 2

# Apply the custom function to each element in the Series
squared_data = data.apply(square)
print("Original Series:")
print(data)
print("\nSeries after applying the custom function:")
print(squared_data)

## 9.Create a new column in a DataFrame that categorizes a numerical column into bins
import pandas as pd

# Sample DataFrame
data = {
    'Values': [1, 15, 22, 5, 18, 30, 12, 8, 25, 20]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Define bins and labels
bins = [0, 10, 20, 30]
labels = ['Low', 'Medium', 'High']

# Create a new column that categorizes the 'Values' column into bins
df['Category'] = pd.cut(df['Values'], bins=bins, labels=labels)
print("\nDataFrame with categorized column:")
print(df)

## 10.Replace all instances of a specific value in a DataFrame column with another value
import pandas as pd

# Sample DataFrame
data = {
    'Column1': [1, 2, 3, 2, 4, 2, 5]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Replace all instances of the value 2 with the value 99
df['Column1'] = df['Column1'].replace(2, 99)
print("\nDataFrame after replacement:")
print(df)
