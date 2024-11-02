### Exploratory Data Analysis (EDA)
## 1.Perform an exploratory data analysis on a dataset by calculating summary statistics (mean, median, mode, standard deviation) for numerical columns
import pandas as pd
import numpy as np
from scipy import stats

# Sample DataFrame
data = {
    'Column1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'Column2': [5, 6, 8, 5, 7, 9, 8, 6, 7, 8],
    'Column3': [11, 21, 31, 41, 51, 61, 71, 81, 91, 101]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Calculate summary statistics
mean = df.mean()
median = df.median()
mode = df.mode().iloc[0]  # The mode can have multiple values, so we take the first
std_dev = df.std()

print("\nMean:")
print(mean)
print("\nMedian:")
print(median)
print("\nMode:")
print(mode)
print("\nStandard Deviation:")
print(std_dev)

## 2.Visualize the distribution of a numerical column using a histogram and a box plot. Identify and explain any outliers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Sample DataFrame
data = {
    'Values': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Create a histogram
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['Values'], bins=10, edgecolor='k')
plt.title('Histogram of Values')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Create a box plot
plt.subplot(1, 2, 2)
plt.boxplot(df['Values'], vert=False)
plt.title('Box Plot of Values')
plt.xlabel('Value')

# Show plots
plt.tight_layout()
plt.show()

## 3.Create a correlation matrix of a DataFrame and visualize it using a heatmap. Interpret the results
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame
data = {
    'Feature1': np.random.randint(1, 100, 50),
    'Feature2': np.random.randint(1, 100, 50),
    'Feature3': np.random.randint(1, 100, 50),
    'Feature4': np.random.randint(1, 100, 50)
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Calculate the correlation matrix
corr_matrix = df.corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()

## 4. Use Pandas to create a scatter plot matrix of multiple columns in a dataset and analyze the relationships between them
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame
data = {
    'Feature1': np.random.randint(1, 100, 50),
    'Feature2': np.random.randint(1, 100, 50),
    'Feature3': np.random.randint(1, 100, 50),
    'Feature4': np.random.randint(1, 100, 50)
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Create a scatter plot matrix
sns.pairplot(df)
plt.suptitle("Scatter Plot Matrix", y=1.02)
plt.show()

## 5.Perform feature engineering on a dataset by creating new features and then visualize their importance using a bar chart
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Sample DataFrame
data = {
    'Feature1': np.random.randint(1, 100, 100),
    'Feature2': np.random.randint(1, 100, 100),
    'Target': np.random.choice([0, 1], 100)
}

df = pd.DataFrame(data)

# Create new features
df['Feature1_plus_Feature2'] = df['Feature1'] + df['Feature2']
df['Feature1_minus_Feature2'] = df['Feature1'] - df['Feature2']
df['Feature1_times_Feature2'] = df['Feature1'] * df['Feature2']
df['Feature1_dividedby_Feature2'] = df['Feature1'] / df['Feature2']

print("DataFrame with new features:")
print(df.head())

# Define features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for the feature importances
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.show()
