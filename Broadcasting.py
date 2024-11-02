### Broadcasting
## 1.Create a 3x3 matrix and add a 1x3 array to each row using broadcasting
import numpy as np

# Create a 3x3 matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original 3x3 matrix:")
print(matrix)

# Create a 1x3 array
array = np.array([10, 20, 30])
print("\n1x3 array:")
print(array)

# Add the 1x3 array to each row of the 3x3 matrix using broadcasting
result = matrix + array
print("\nResultant matrix:")
print(result)

## 2.Multiply a 1D array of 5 elements by a scalar value using broadcasting
import numpy as np

# Create a 1D array with 5 elements
array = np.array([1, 2, 3, 4, 5])
print("Original array:")
print(array)

# Multiply the array by a scalar value
scalar = 10
result = array * scalar
print("\nResult after multiplying by scalar value:")
print(result)

## 3.Subtract a 3x1 column vector from a 3x3 matrix using broadcasting
import numpy as np

# Create a 3x3 matrix
matrix = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
print("Original 3x3 matrix:")
print(matrix)

# Create a 3x1 column vector
column_vector = np.array([[5], [15], [25]])
print("\n3x1 column vector:")
print(column_vector)

# Subtract the column vector from the matrix using broadcasting
result = matrix - column_vector
print("\nResultant matrix:")
print(result)

## 4.Add a scalar to a 3D array and demonstrate how broadcasting works across all dimensions
import numpy as np

# Create a 3D array (2x3x3)
array_3d = np.random.randint(0, 10, (2, 3, 3))
print("Original 3D array:")
print(array_3d)

# Define a scalar value to add
scalar = 5
print("\nScalar to add:", scalar)

# Add the scalar to the 3D array using broadcasting
result = array_3d + scalar
print("\nResultant 3D array after adding scalar:")
print(result)

## 5.Create two arrays of shapes (4, 1) and (1, 5) and add them using broadcasting
import numpy as np

# Create two arrays
array1 = np.random.randint(0, 10, (4, 1))
array2 = np.random.randint(0, 10, (1, 5))

print("Array 1:")
print(array1)
print("\nArray 2:")
print(array2)

# Add the arrays using broadcasting
result = array1 + array2
print("\nResultant array after broadcasting:")
print(result)
