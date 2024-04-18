import numpy as np

def gaussian_elimination(A):
    dim0, dim1 = A.shape
    swap_matrix = np.eye(dim0)
    scale_matrix = np.eye(dim0)

    for k in range(min(dim0, dim1)):
        pivot_row = np.argmax(np.abs(A[k:, k])) + k
        if A[pivot_row, k] == 0:
            continue

        if pivot_row != k:
            A[[k, pivot_row], :] = A[[pivot_row, k], :]
            swap_matrix[[k, pivot_row], :] = swap_matrix[[pivot_row, k], :]

        scale_factor = 1 / A[k, k]
        A[k, :] *= scale_factor
        scale_matrix[k, k] = scale_factor

        for i in range(dim0):
            if i != k:
                factor = A[i, k]
                A[i, :] -= factor * A[k, :]
                scale_matrix[i, k] = -factor

    return swap_matrix, scale_matrix, A

# Example usage:
A = np.array([[2, 1, -1, 8],
              [-3, -1, 2, -11],
              [-2, 1, 2, -3]], dtype=float)

swap_matrix, scale_matrix, echelon_form = gaussian_elimination(A)

print("Swap matrix:")
print(swap_matrix)
print("\nScale matrix:")
print(scale_matrix)
print("\nReduced echelon form:")
print(echelon_form)
