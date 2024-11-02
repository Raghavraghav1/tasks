### Vectorized Operations
## 1.Compute the square root of each element in a 2D array using vectorized operations
import numpy as np

# Create a 2D array
array = np.random.randint(1, 10, (3, 3))
print("Original 2D array:")
print(array)

# Compute the square root of each element
sqrt_array = np.sqrt(array)
print("\nSquare root of each element:")
print(sqrt_array)

## 2.Calculate the dot product of two 1D arrays of size 5
import numpy as np

# Create two 1D arrays of size 5
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([5, 4, 3, 2, 1])

# Calculate the dot product
dot_product = np.dot(array1, array2)
print("Dot product:", dot_product)

## 3.Perform element-wise comparison of two 1D arrays and return an array of boolean values indicating where the first array has larger elements
import numpy as np

# Create two 1D arrays
array1 = np.array([10, 15, 20, 25, 30])
array2 = np.array([5, 15, 25, 20, 10])

# Perform element-wise comparison
comparison = array1 > array2
print("Element-wise comparison:")
print(comparison)

## 4.Create a 2D array and apply a vectorized operation to double the value of each element
import numpy as np

# Create a 2D array
array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original 2D array:")
print(array)

# Apply a vectorized operation to double the value of each element
doubled_array = array * 2
print("\nDoubled 2D array:")
print(doubled_array)

## 5.Create a 1D array of 100 random integers and compute the sum of all even numbers using vectorized operations
import numpy as np

# Create a 1D array of 100 random integers
array = np.random.randint(1, 101, 100)
print("Original array:")
print(array)

# Compute the sum of all even numbers
even_sum = np.sum(array[array % 2 == 0])
print("\nSum of all even numbers:")
print(even_sum)
