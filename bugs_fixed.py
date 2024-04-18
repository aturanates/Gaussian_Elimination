import numpy as np

def eliminateCol(A, k=0):
    # guarantee that A is float type
    if A.dtype.kind != 'f' and A.dtype.kind != 'c':
        return None, None
    # load dimensions
    dim0, dim1 = np.shape(A)
    # initialize outputs
    swap = np.identity(dim0, dtype=float)
    oper = np.identity(dim0, dtype=float)
    # k < dim1
    k = k % dim1
    # if the column k is a zero column, return
    if not np.any(A[:, k]):
        return swap, oper

    i0 = 0
    if k != 0:
        # find a row i0 whose leftmost k-1 entries are all zero
        while i0 < dim0:
            if not np.any(A[i0, 0:k]):
                break
            i0 += 1
    # find a nonzero element A[i, k] to use as a pivot
    i = i0
    while i < dim0:
        if A[i, k] != 0:
            break
        i += 1
    if i == dim0:
        return swap, oper
    # swap the row i with the row i0
    if i != i0:
        A[[i0, i], :] = A[[i, i0], :]
        swap[[i0, i], :] = swap[[i, i0], :]
    # row i0 is the pivot row and A[j, k] is zero for i0 < j <= i
    i = i + 1
    while i < dim0:
        if A[i, k] != 0:
            quot = A[i, k] / A[i0, k]
            A[i, :] = A[i, :] - quot * A[i0, :]
            oper[i, i0] = -quot
        i += 1
    return swap, oper

def eliminateRow(A, k=-1):
    # guarantee that A is float type
    if A.dtype.kind != 'f' and A.dtype.kind != 'c':
        return None
    # load dimensions
    dim0, dim1 = np.shape(A)
    # initialize outputs
    oper = np.identity(dim0, dtype=float)
    # k < dim0
    k = k % dim0
    # if the row k is a zero row, return
    if k == 0 or not np.any(A[k, :]):
        return oper

    # find a nonzero element A[k, j] to use as a pivot
    j = 0
    while j < dim1:
        if A[k, j] != 0:
            break
        j += 1
    i = k - 1
    while i >= 0:
        if A[i, j] != 0:
            quot = A[i, j] / A[k, j]
            A[i, :] = A[i, :] - quot * A[k, :]
            oper[i, k] = -quot
        i -= 1
    return oper

def eliminateScale(A):
    # guarantee that A is float type
    if A.dtype.kind != 'f' and A.dtype.kind != 'c':
        return None
    # load dimensions
    dim0, dim1 = np.shape(A)
    # initialize outputs
    oper = np.identity(dim0, dtype=float)
    # find pivots
    i = 0
    while i < dim0:
        j = 0
        while j < dim1:
            if A[i, j] == 0:
                j += 1
                continue
            else:
                oper[i, i] = 1 / A[i, j]
                A[i, j:] = A[i, j:] / A[i, j]
                break
            j += 1
        i += 1
    return oper

def gaussianElimination(A):
    # guarantee that A is float type
    if A.dtype.kind != 'f' and A.dtype.kind != 'c':
        return None
    # load dimensions
    dim0, dim1 = np.shape(A)
    # initialize outputs
    swap_mat = np.identity(dim0, dtype=float)
    scale_mat = np.identity(dim0, dtype=float)
    for k in range(min(dim0, dim1)):
        swap, oper = eliminateCol(A, k)
        swap_mat = swap @ swap_mat
        A = oper @ A
        scale = eliminateScale(A)
        scale_mat = scale @ scale_mat
        for i in range(k, dim0):
            oper = eliminateRow(A, i)
            A = oper @ A
    return swap_mat, scale_mat, A

# Example usage:
A = np.array([[2, 1, -1, 8],
              [-3, -1, 2, -11],
              [-2, 1, 2, -3]], dtype=float)

swap_matrix, scale_matrix, echelon_form = gaussianElimination(A)

print("Swap matrix:")
print(swap_matrix)
print("\nScale matrix:")
print(scale_matrix)
print("\nReduced echelon form:")
print(echelon_form)
