### Array Manipulation
## 1.Reshape a 1D array of 12 elements into a 3x4 matrix.
import numpy as np

# Create a 1D array with 12 elements
array = np.arange(1, 13)  # This will create an array [1, 2, 3, ..., 12]
print("Original 1D array:")
print(array)

# Reshape the array into a 3x4 matrix
reshaped_array = array.reshape(3, 4)
print("\nReshaped 3x4 matrix:")
print(reshaped_array)

## 2.Flatten a 3x3 matrix into a 1D array.
import numpy as np

# Create a 3x3 matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original 3x3 matrix:")
print(matrix)

# Flatten the matrix into a 1D array
flattened_array = matrix.flatten()
print("\nFlattened 1D array:")
print(flattened_array)

## 3.Stack two 3x3 matrices horizontally and vertically.
import numpy as np

# Create two 3x3 matrices with random integers between 0 and 10
matrix1 = np.random.randint(0, 11, (3, 3))
matrix2 = np.random.randint(0, 11, (3, 3))

print("Matrix 1:")
print(matrix1)
print("\nMatrix 2:")
print(matrix2)

# Stack the matrices horizontally
horizontal_stack = np.hstack((matrix1, matrix2))
print("\nHorizontally Stacked:")
print(horizontal_stack)

# Stack the matrices vertically
vertical_stack = np.vstack((matrix1, matrix2))
print("\nVertically Stacked:")
print(vertical_stack)

## 4.Concatenate two arrays of different sizes along a new axis.
import numpy as np

# Create two arrays of different sizes
array1 = np.array([1, 2, 3])
array2 = np.array([4, 5, 6, 7])

# Concatenate the arrays along a new axis
concatenated_array = np.concatenate([array1[:, np.newaxis], array2[:, np.newaxis][:3]], axis=1)
print(concatenated_array)

## 5.Transpose a 3x2 matrix and then reshape it to have 3 rows and 2 columns.
import numpy as np

# Create a 3x2 matrix
matrix = np.array([[1, 2], [3, 4], [5, 6]])
print("Original 3x2 matrix:")
print(matrix)

# Transpose the matrix
transposed_matrix = matrix.T
print("\nTransposed matrix:")
print(transposed_matrix)

# Reshape the transposed matrix to have 3 rows and 2 columns
reshaped_matrix = transposed_matrix.reshape(3, 2)
print("\nReshaped matrix:")
print(reshaped_matrix)
