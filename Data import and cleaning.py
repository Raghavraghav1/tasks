### Data Import and Cleaning
## 1.Import a CSV file into a Pandas DataFrame. Identify and drop rows with missing values
import pandas as pd

# Import CSV file into a Pandas DataFrame
df = pd.read_csv('your_file.csv')
print("Original DataFrame:")
print(df)

# Identify and drop rows with missing values
df_cleaned = df.dropna()
print("\nDataFrame after dropping rows with missing values:")
print(df_cleaned)

## 2.Load a dataset and replace missing numerical values with the mean of the column
import pandas as pd

# Load the dataset into a Pandas DataFrame
df = pd.read_csv('your_file.csv')
print("Original DataFrame:")
print(df)

# Replace missing numerical values with the mean of the column
df_filled = df.apply(lambda col: col.fillna(col.mean()) if col.dtype in ['float64', 'int64'] else col)
print("\nDataFrame after replacing missing values with the column mean:")
print(df_filled)

## 3.Replace missing categorical values with the mode of the column in a DataFrame
import pandas as pd

# Sample DataFrame with missing categorical values
data = {
    'Category': ['A', 'B', None, 'B', 'C', 'C', None, 'A']
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Replace missing categorical values with the mode of the column
df['Category'] = df['Category'].fillna(df['Category'].mode()[0])
print("\nDataFrame after replacing missing values with the mode:")
print(df)

### Data Transformation
## 1. Create a new column in a DataFrame that is the sum of two existing columns using NumPy vectorized operations
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Column1': [1, 2, 3, 4, 5],
    'Column2': [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Create a new column that is the sum of 'Column1' and 'Column2'
df['Sum'] = np.add(df['Column1'], df['Column2'])
print("\nDataFrame with new 'Sum' column:")
print(df)

## 2.Apply a mathematical function (e.g., square root) to all elements of a numerical column using NumPy
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Values': [1, 4, 9, 16, 25]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Apply the square root function to all elements of the 'Values' column
df['Sqrt_Values'] = np.sqrt(df['Values'])
print("\nDataFrame after applying square root to 'Values' column:")
print(df)

## 3.Normalize a numerical column in a DataFrame using MinMaxScaler from sklearn.preprocessing and explain the transformation process
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Sample DataFrame
data = {
    'Values': [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the data
df['Normalized_Values'] = scaler.fit_transform(df[['Values']])
print("\nDataFrame after MinMax scaling:")
print(df)

### Merging and Joining Datasets
## 1.Merge two DataFrames based on a common key and fill any missing values in the resulting DataFrame
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

# Merge the DataFrames based on the common key
merged_df = pd.merge(df1, df2, on='Key', how='outer')
print("\nMerged DataFrame:")
print(merged_df)

# Fill any missing values with a specified value (e.g., 0)
merged_df_filled = merged_df.fillna(0)
print("\nMerged DataFrame after filling missing values:")
print(merged_df_filled)

## 2.Perform a left join on two DataFrames with different keys and handle missing data in the result
import pandas as pd

# Sample DataFrames
data1 = {
    'Key1': ['A', 'B', 'C'],
    'Value1': [1, 2, 3]
}

data2 = {
    'Key2': ['B', 'C', 'D'],
    'Value2': [4, 5, 6]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# Perform a left join on the DataFrames
merged_df = pd.merge(df1, df2, left_on='Key1', right_on='Key2', how='left')
print("\nMerged DataFrame (Left Join):")
print(merged_df)

# Handle missing data by filling with a specified value (e.g., 0)
merged_df_filled = merged_df.fillna(0)
print("\nMerged DataFrame after filling missing values:")
print(merged_df_filled)

## 3.Concatenate two DataFrames along the columns and handle any duplicate column names
import pandas as pd

# Sample DataFrames
data1 = {
    'A': [1, 2, 3],
    'B': [4, 5, 6]
}

data2 = {
    'B': [7, 8, 9],
    'C': [10, 11, 12]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# Concatenate DataFrames along the columns
concatenated_df = pd.concat([df1, df2], axis=1)

# Handle duplicate column names by appending a suffix
concatenated_df = df1.add_suffix('_df1').join(df2.add_suffix('_df2'))

print("\nConcatenated DataFrame with handled duplicate column names:")
print(concatenated_df)

### Grouping and Aggregation
## 1.Group a DataFrame by a categorical column and calculate the mean and standard deviation of a numerical column for each group
import pandas as pd

# Sample DataFrame
data = {
    'Category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'Values': [10, 20, 30, 40, 50, 60, 70, 80]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Group by 'Category' and calculate mean and standard deviation of 'Values'
grouped_df = df.groupby('Category').agg({'Values': ['mean', 'std']})
print("\nGrouped DataFrame with mean and standard deviation:")
print(grouped_df)

## 2.Use groupby() to calculate the sum of a column for each group, then apply a NumPy function to the grouped results
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
    'Values': [10, 20, 30, 40, 50, 60, 70, 80]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Group by 'Category' and calculate the sum of 'Values'
grouped_sum = df.groupby('Category')['Values'].sum()
print("\nGrouped sum of 'Values':")
print(grouped_sum)

# Apply a NumPy function (e.g., square root) to the grouped results
grouped_sum_sqrt = grouped_sum.apply(np.sqrt)
print("\nGrouped sum with square root applied:")
print(grouped_sum_sqrt)

## 3.Create a pivot table from a DataFrame that groups data by two categorical columns and summarizes a numerical column using NumPy operations
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Category1': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A'],
    'Category2': ['X', 'Y', 'X', 'Y', 'X', 'Y', 'Y', 'X', 'Y', 'X'],
    'Values': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Create a pivot table
pivot_table = pd.pivot_table(df, 
                             values='Values', 
                             index=['Category1', 'Category2'], 
                             aggfunc={'Values': [np.sum, np.mean, np.std]})

print("\nPivot Table with NumPy operations:")
print(pivot_table)

### Array Operations and Manipulation
## 1.Create a NumPy array from a DataFrame column and perform element-wise operations on the array
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Values': [1, 2, 3, 4, 5]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Create a NumPy array from the 'Values' column
array = df['Values'].to_numpy()
print("\nNumPy array from 'Values' column:")
print(array)

# Perform element-wise operations (e.g., add 10 to each element)
array_plus_10 = array + 10
print("\nArray after adding 10 to each element:")
print(array_plus_10)

# Perform element-wise square of each element
array_squared = np.square(array)
print("\nArray after squaring each element:")
print(array_squared)

## 2.Reshape a NumPy array and assign it back to a new DataFrame column
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Values': [1, 2, 3, 4, 5, 6]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Create a NumPy array from the 'Values' column
array = df['Values'].to_numpy()
print("\nNumPy array from 'Values' column:")
print(array)

# Reshape the array (for example, to 2x3)
reshaped_array = array.reshape(2, 3)
print("\nReshaped array (2x3):")
print(reshaped_array)

# Flatten the reshaped array to assign it back to a DataFrame column
flattened_array = reshaped_array.flatten()
df['New_Values'] = flattened_array
print("\nDataFrame with new 'New_Values' column:")
print(df)

## 3.Use NumPy to filter a DataFrame for rows where a numerical column's values are above a certain threshold
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Values': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Define the threshold
threshold = 5

# Use NumPy to filter rows where 'Values' column is above the threshold
filtered_df = df[np.array(df['Values'] > threshold)]
print("\nFiltered DataFrame:")
print(filtered_df)

### Broadcasting and Vectorized Operations
## 1.Broadcast a NumPy array across a DataFrame column to perform a vectorized operation
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Values': [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Create a NumPy array for broadcasting
array = np.array([1, 2, 3, 4, 5])

# Broadcast the array across the 'Values' column and perform an operation (e.g., addition)
df['Broadcasted_Sum'] = df['Values'] + array
print("\nDataFrame after broadcasting and addition operation:")
print(df)

## 2.Create a new column in a DataFrame that results from a vectorized operation on multiple columns using NumPy
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Column1': [10, 20, 30, 40, 50],
    'Column2': [5, 4, 3, 2, 1]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Perform a vectorized operation (e.g., add and multiply) on multiple columns using NumPy
df['New_Column'] = np.add(df['Column1'], df['Column2']) * np.multiply(df['Column1'], df['Column2'])
print("\nDataFrame after vectorized operation:")
print(df)

## 3.Demonstrate broadcasting by subtracting the mean of each row from the row's elements in a DataFrame
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [5, 4, 3, 2, 1],
    'C': [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Subtract the mean of each row from the row's elements using broadcasting
row_means = df.mean(axis=1).values.reshape(-1, 1)
broadcasted_df = df - row_means
print("\nDataFrame after subtracting row means:")
print(broadcasted_df)

### Linear Algebra with NumPy
## 1.Solve a system of linear equations using NumPy, and store the solution in a DataFrame
import numpy as np
import pandas as pd

# Coefficients of the equations
A = np.array([[2, -3], [4, 1]])

# Constants on the right-hand side
B = np.array([8, 10])

# Solve the system of equations
solution = np.linalg.solve(A, B)
print("Solution:")
print("x =", solution[0])
print("y =", solution[1])

# Store the solution in a DataFrame
solution_df = pd.DataFrame({'Variable': ['x', 'y'], 'Value': solution})
print("\nSolution DataFrame:")
print(solution_df)

## 2.Compute the dot product of two columns from a DataFrame using NumPy
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Column1': [1, 2, 3, 4, 5],
    'Column2': [10, 20, 30, 40, 50]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Compute the dot product of 'Column1' and 'Column2'
dot_product = np.dot(df['Column1'], df['Column2'])
print("\nDot product of 'Column1' and 'Column2':", dot_product)

## 3.Perform matrix multiplication on two DataFrames treated as matrices and store the result in a new DataFrame
import pandas as pd
import numpy as np

# Sample DataFrames
data1 = {
    'A': [1, 2],
    'B': [3, 4]
}

data2 = {
    'C': [5, 6],
    'D': [7, 8]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)

print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# Perform matrix multiplication
result_matrix = np.dot(df1, df2)
result_df = pd.DataFrame(result_matrix, columns=['C', 'D'])

print("\nResult DataFrame after matrix multiplication:")
print(result_df)

### Handling Missing Data
## 1.Interpolate missing values in a DataFrame using a linear method with NumPy and Pandas
import pandas as pd
import numpy as np

# Sample DataFrame with missing values
data = {
    'Values': [1, 2, np.nan, 4, np.nan, 6, 7]
}

df = pd.DataFrame(data)
print("Original DataFrame with missing values:")
print(df)

# Interpolate missing values using a linear method
df['Values'] = df['Values'].interpolate(method='linear')
print("\nDataFrame after linear interpolation:")
print(df)

## 2.Use a mask created with NumPy to fill missing values in a DataFrame with a specific value
import pandas as pd
import numpy as np

# Sample DataFrame with missing values
data = {
    'Values': [1, 2, np.nan, 4, np.nan, 6, 7]
}

df = pd.DataFrame(data)
print("Original DataFrame with missing values:")
print(df)

# Create a mask to identify missing values
mask = np.isnan(df['Values'])

# Fill missing values with a specific value (e.g., 99)
df['Values'][mask] = 99
print("\nDataFrame after filling missing values with 99:")
print(df)

## 3.Identify and replace outliers in a DataFrame column with the median value using a NumPy mask
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Values': [1, 2, 3, 4, 5, 6, 7, 8, 100, 101, 9, 10]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Calculate the median value of the column
median_value = df['Values'].median()

# Calculate the z-scores to identify outliers
z_scores = np.abs((df['Values'] - df['Values'].mean()) / df['Values'].std())
print("\nZ-scores:")
print(z_scores)

# Create a mask to identify outliers (considering a threshold, e.g., z-score > 2)
threshold = 2
outliers_mask = z_scores > threshold
print("\nOutliers mask:")
print(outliers_mask)

# Replace outliers with the median value
df.loc[outliers_mask, 'Values'] = median_value
print("\nDataFrame after replacing outliers with median value:")
print(df)

### Advanced Data Analysis
## 1.Use a combination of groupby() and NumPy operations to analyze trends in a multi-level categorical dataset
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'Category1': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'Category2': ['X', 'X', 'Y', 'Y', 'Z', 'Z', 'X', 'Y', 'Y'],
    'Values': [10, 15, 20, 25, 30, 35, 40, 45, 50]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Group by 'Category1' and 'Category2' and calculate the mean and sum of 'Values'
grouped = df.groupby(['Category1', 'Category2'])['Values'].agg(['mean', 'sum'])
print("\nGrouped DataFrame with mean and sum:")
print(grouped)

# Calculate the percentage change between groups
grouped['Percent_Change'] = grouped['sum'].pct_change() * 100
print("\nGrouped DataFrame with percent change:")
print(grouped)

# Fill NaN values in 'Percent_Change' with 0 (if any)
grouped['Percent_Change'] = grouped['Percent_Change'].fillna(0)
print("\nGrouped DataFrame after filling NaN values in percent change:")
print(grouped)

## 2.Create a summary DataFrame that includes the correlation matrix of numerical columns using both Pandas and NumPy
import pandas as pd
import numpy as np

# Sample DataFrame with numerical columns
data = {
    'Column1': np.random.randint(1, 100, 10),
    'Column2': np.random.randint(1, 100, 10),
    'Column3': np.random.randint(1, 100, 10)
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Calculate the correlation matrix
corr_matrix = df.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Create a summary DataFrame that includes the correlation matrix
summary_df = corr_matrix.copy()
summary_df['Mean'] = df.mean()
summary_df['Standard Deviation'] = df.std()
print("\nSummary DataFrame:")
print(summary_df)

## 3.Perform a rolling mean calculation on a time series dataset and visualize the result using Matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample time series data
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
data = np.random.randn(100).cumsum()  # Generate random walk data

# Create a DataFrame
df = pd.DataFrame(data, index=dates, columns=['Value'])
print("Original DataFrame:")
print(df.head())

# Calculate the rolling mean with a window size of 7 days
df['Rolling_Mean'] = df['Value'].rolling(window=7).mean()
print("\nDataFrame with Rolling Mean:")
print(df.head(10))

# Visualize the result
plt.figure(figsize=(12, 6))
plt.plot(df['Value'], label='Original')
plt.plot(df['Rolling_Mean'], label='Rolling Mean (7 days)', color='red')
plt.title('Time Series with Rolling Mean')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

### DataFrame and Array Manipulation
## 1.Convert a DataFrame into a NumPy array, perform an operation, and convert it back to a DataFrame
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5],
    'B': [6, 7, 8, 9, 10]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Convert the DataFrame to a NumPy array
array = df.to_numpy()
print("\nNumPy array:")
print(array)

# Perform an operation (e.g., add 10 to each element)
array = array + 10
print("\nNumPy array after adding 10 to each element:")
print(array)

# Convert the NumPy array back to a DataFrame
new_df = pd.DataFrame(array, columns=df.columns)
print("\nNew DataFrame:")
print(new_df)

## 2. Use NumPy to generate a DataFrame with random values, then apply a condition to filter rows based on multiple criteria
import pandas as pd
import numpy as np

# Generate a DataFrame with random values
np.random.seed(42)  # For reproducibility
data = np.random.randint(1, 101, size=(10, 3))  # 10 rows and 3 columns with random integers between 1 and 100
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print("Original DataFrame:")
print(df)

# Apply a condition to filter rows (e.g., A > 50 and B < 50)
filtered_df = df[(df['A'] > 50) & (df['B'] < 50)]
print("\nFiltered DataFrame (A > 50 and B < 50):")
print(filtered_df)

## 3.Create a DataFrame where each element is the result of applying a custom NumPy function to corresponding elements in two NumPy arrays
import pandas as pd
import numpy as np

# Define a custom NumPy function
def custom_function(x, y):
    return x ** 2 + y ** 2

# Create two NumPy arrays
array1 = np.random.randint(1, 10, size=(5, 3))
array2 = np.random.randint(1, 10, size=(5, 3))

print("Array 1:")
print(array1)
print("\nArray 2:")
print(array2)

# Apply the custom function to the corresponding elements of the two arrays
result_array = custom_function(array1, array2)

# Create a DataFrame from the resulting array
df = pd.DataFrame(result_array, columns=['A', 'B', 'C'])
print("\nDataFrame after applying custom function:")
print(df)

### Data Reshaping and Analysis
## 1.Use NumPy's reshape to change the shape of an array extracted from a DataFrame and analyze the reshaped data
import pandas as pd
import numpy as np

# Sample DataFrame
data = {
    'A': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'B': [9, 8, 7, 6, 5, 4, 3, 2, 1]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Extract an array from the DataFrame
array = df.to_numpy()
print("\nExtracted Array:")
print(array)

# Reshape the array (for example, from 9x2 to 3x6)
reshaped_array = array.reshape(3, 6)
print("\nReshaped Array (3x6):")
print(reshaped_array)

# Analyze the reshaped data
mean_values = np.mean(reshaped_array, axis=0)
std_values = np.std(reshaped_array, axis=0)

print("\nMean values of reshaped array columns:")
prin
