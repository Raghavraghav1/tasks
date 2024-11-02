### Linear Algebra
## 1.Create a 3x3 matrix and find its determinant
import numpy as np

# Create a 3x3 matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original 3x3 matrix:")
print(matrix)

# Find the determinant of the matrix
determinant = np.linalg.det(matrix)
print("\nDeterminant of the matrix:")
print(determinant)

## 2.Given a 2x2 matrix, compute its inverse and verify by multiplying with the original matrix to obtain the identity matrix
import numpy as np

# Create a 2x2 matrix
matrix = np.array([[4, 7], [2, 6]])
print("Original matrix:")
print(matrix)

# Compute the inverse of the matrix
inverse_matrix = np.linalg.inv(matrix)
print("\nInverse matrix:")
print(inverse_matrix)

# Verify by multiplying the original matrix with its inverse
identity_matrix = np.dot(matrix, inverse_matrix)
print("\nIdentity matrix (verification):")
print(identity_matrix)

## 3.Calculate the eigenvalues and eigenvectors of a 2x2 matrix
import numpy as np

# Create a 2x2 matrix
matrix = np.array([[4, -2], [1, 1]])
print("Original matrix:")
print(matrix)

# Calculate the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors:")
print(eigenvectors)

## 4.Solve the system of linear equations 2x + 3y = 5 and 4x + 6y = 10 using NumPy
import numpy as np

# Coefficients of the equations
A = np.array([[2, 3], [4, 6]])

# Constants on the right-hand side
B = np.array([5, 10])

try:
    # Solve the system of equations
    solution = np.linalg.solve(A, B)
    print("Solution:")
    print("x =", solution[0])
    print("y =", solution[1])
except np.linalg.LinAlgError:
    print("The system of equations has no unique solution.")

## 5.Perform Singular Value Decomposition (SVD) on a 3x3 matrix and reconstruct the original matrix using the SVD components
import numpy as np

# Create a 3x3 matrix
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original matrix:")
print(matrix)

# Perform Singular Value Decomposition
U, S, Vt = np.linalg.svd(matrix)
print("\nU matrix:")
print(U)
print("\nSingular values:")
print(S)
print("\nVt matrix:")
print(Vt)

# Reconstruct the original matrix
S_matrix = np.diag(S)  # Convert the singular values into a diagonal matrix
reconstructed_matrix = U @ S_matrix @ Vt
print("\nReconstructed matrix:")
print(reconstructed_matrix)
