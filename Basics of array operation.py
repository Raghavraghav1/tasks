### Basic Array Operations
## 1. Create a 3x3 array filled with random integers between 0 and 10. Calculate the sum, mean, and standard deviation of the array.
import numpy as np
# Create a 3x3 array with random integers between 0 and 10
array = np.random.randint(0, 11, (3, 3))
print("Array:")
print(array)

# Calculate the sum of the array
sum_array = np.sum(array)
print("\nSum:", sum_array)

# Calculate the mean of the array
mean_array = np.mean(array)
print("Mean:", mean_array)

# Calculate the standard deviation of the array
std_dev_array = np.std(array)
print("Standard Deviation:", std_dev_array)

## 2. Create a 1D array of 10 elements and compute the cumulative sum of the elements. 
import numpy as np

# Create a 1D array with 10 elements
array = np.random.randint(0, 10, 10)
print("Array:")
print(array)

# Compute the cumulative sum of the elements
cumulative_sum = np.cumsum(array)
print("\nCumulative Sum:")
print(cumulative_sum)

## 3.Generate two 2x3 arrays with random integers and perform element-wise addition, subtraction, multiplication, and division.
import numpy as np

# Generate two 2x3 arrays with random integers between 0 and 10
array1 = np.random.randint(0, 11, (2, 3))
array2 = np.random.randint(0, 11, (2, 3))

print("Array 1:")
print(array1)
print("\nArray 2:")
print(array2)

# Element-wise addition
addition = np.add(array1, array2)
print("\nElement-wise Addition:")
print(addition)

# Element-wise subtraction
subtraction = np.subtract(array1, array2)
print("\nElement-wise Subtraction:")
print(subtraction)

# Element-wise multiplication
multiplication = np.multiply(array1, array2)
print("\nElement-wise Multiplication:")
print(multiplication)

# Element-wise division
division = np.divide(array1, array2)
print("\nElement-wise Division:")
print(division)

## 4.Create a 4x4 identity matrix.
import numpy as np

# Create a 4x4 identity matrix
identity_matrix = np.eye(4)
print(identity_matrix)

## 5.Given an array a = np.array([5, 10, 15, 20, 25]), divide each element by 5 using broadcasting.
import numpy as np

# Given array
a = np.array([5, 10, 15, 20, 25])

# Divide each element by 5 using broadcasting
result = a / 5
print(result)
