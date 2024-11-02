### NumPy
## 1.Create a NumPy array of random numbers and normalize the data using min-max scaling
import numpy as np

# Create a NumPy array of random numbers
array = np.random.randint(1, 100, 10)
print("Original array:")
print(array)

# Perform min-max scaling
min_val = np.min(array)
max_val = np.max(array)
normalized_array = (array - min_val) / (max_val - min_val)
print("\nNormalized array using min-max scaling:")
print(normalized_array)

## 2.Generate a 5x5 matrix with random integers and replace all occurrences of a specific value with another
import numpy as np

# Generate a 5x5 matrix with random integers between 0 and 10
matrix = np.random.randint(0, 11, (5, 5))
print("Original matrix:")
print(matrix)

# Specify the value to replace and the new value
value_to_replace = 5
new_value = 99

# Replace all occurrences of the specified value with the new value
matrix[matrix == value_to_replace] = new_value
print("\nModified matrix:")
print(matrix)

## 3.Create two NumPy arrays and perform element-wise addition, subtraction, multiplication, and division
import numpy as np

# Create two arrays with random integers
array1 = np.random.randint(1, 10, size=(3, 3))
array2 = np.random.randint(1, 10, size=(3, 3))

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

## 4.Solve the system of linear equations: \(3x + 4y = 7\) and \(5x + 2y = 8\) using NumPy
import numpy as np

# Coefficients of the equations
A = np.array([[3, 4], [5, 2]])

# Constants on the right-hand side
B = np.array([7, 8])

# Solve the system of equations
solution = np.linalg.solve(A, B)
print("Solution:")
print("x =", solution[0])
print("y =", solution[1])

## 5.Implement broadcasting to perform an operation between a 3x3 matrix and a 1D array
import numpy as np

# Create a 3x3 matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original 3x3 matrix:")
print(matrix)

# Create a 1D array
array = np.array([10, 20, 30])
print("\n1D array:")
print(array)

# Perform broadcasting to add the 1D array to each row of the 3x3 matrix
result = matrix + array
print("\nResult after broadcasting:")
print(result)

## 6.Create a 3x3 identity matrix using NumPy
import numpy as np

# Create a 3x3 identity matrix
identity_matrix = np.eye(3)
print(identity_matrix)

## 7.Perform matrix multiplication between two 2D arrays
import numpy as np

# Create two 2D arrays
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

print("Matrix 1:")
print(matrix1)
print("\nMatrix 2:")
print(matrix2)

# Perform matrix multiplication
result = np.dot(matrix1, matrix2)
print("\nResult of matrix multiplication:")
print(result)

## 8.Calculate the dot product and cross product of two vectors using NumPy
import numpy as np

# Create two vectors
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

# Calculate the dot product
dot_product = np.dot(vector1, vector2)
print("Dot product:", dot_product)

# Calculate the cross product
cross_product = np.cross(vector1, vector2)
print("Cross product:", cross_product)

## 9.Generate a NumPy array of 20 random integers and find the unique elements in the array
import numpy as np

# Generate a NumPy array of 20 random integers
array = np.random.randint(0, 50, 20)
print("Original array:")
print(array)

# Find the unique elements in the array
unique_elements = np.unique(array)
print("\nUnique elements:")
print(unique_elements)

## 10.Implement a function using NumPy that returns the inverse of a given matrix
import numpy as np

def inverse_matrix(matrix):
    try:
        # Compute the inverse of the matrix
        inverse = np.linalg.inv(matrix)
        return inverse
    except np.linalg.LinAlgError:
        return "Matrix is singular and cannot be inverted."

# Example usage
matrix = np.array([[1, 2], [3, 4]])
print("Original matrix:")
print(matrix)

inverse = inverse_matrix(matrix)
print("\nInverse matrix:")
print(inverse)
