### Indexing and Slicing
## 1.Given a 1D array of 15 elements, extract elements at positions 2 to 10 with a step of 2
import numpy as np

# Create a 1D array with 15 elements
array = np.arange(1, 16)
print("Original array:")
print(array)

# Extract elements at positions 2 to 10 with a step of 2
extracted_elements = array[2:11:2]
print("\nExtracted elements:")
print(extracted_elements)

## 2.Create a 5x5 matrix and extract the sub-matrix containing elements from rows 1 to 3 and columns 2 to 4
import numpy as np

# Create a 5x5 matrix
matrix = np.random.randint(1, 101, (5, 5))
print("Original 5x5 matrix:")
print(matrix)

# Extract the sub-matrix from rows 1 to 3 and columns 2 to 4
sub_matrix = matrix[1:4, 2:5]
print("\nExtracted sub-matrix (rows 1 to 3, columns 2 to 4):")
print(sub_matrix)

## 3.Replace all elements in a 1D array greater than 10 with the value 10
import numpy as np

# Create a 1D array
array = np.array([5, 12, 9, 15, 7, 18, 3, 10, 22, 1])
print("Original array:")
print(array)

# Replace elements greater than 10 with 10
array[array > 10] = 10
print("\nModified array:")
print(array)

## 4.Use fancy indexing to select elements from a 1D array at positions [0, 2, 4, 6]
import numpy as np

# Create a 1D array
array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("Original array:")
print(array)

# Use fancy indexing to select elements at positions [0, 2, 4, 6]
selected_elements = array[[0, 2, 4, 6]]
print("\nSelected elements at positions [0, 2, 4, 6]:")
print(selected_elements)

## 5.Create a 1D array of 10 elements and reverse it using slicing
import numpy as np

# Create a 1D array with 10 elements
array = np.arange(10)
print("Original array:")
print(array)

# Reverse the array using slicing
reversed_array = array[::-1]
print("\nReversed array:")
print(reversed_array)
