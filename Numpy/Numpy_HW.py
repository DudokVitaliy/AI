import numpy as np

print("=== N1 ===")

arr1 = np.random.randint(0, 51, 20)
print("Array:", arr1)

threshold = int(input("Enter threshold value: "))

count = np.sum(arr1 > threshold)
print("Count elements > threshold:", count)

max_val = np.max(arr1)
max_index = np.argmax(arr1)
print("Max:", max_val)
print("First occurrence position:", max_index)
sorted_arr = np.sort(arr1)[::-1]
print("Sorted array (descending):", sorted_arr)

print("\n=== N2 ===")

low = int(input("Enter lower limit: "))
high = int(input("Enter upper limit: "))

matrix2 = np.random.randint(low, high + 1, (5, 5))
print("Matrix:\n", matrix2)

diag = np.diag(matrix2)
print("Main diagonal:", diag)

diag_sum = np.sum(diag)
print("Diagonal sum:", diag_sum)

matrix2_mod = matrix2.copy()
matrix2_mod[np.triu_indices(5, 1)] = 0
print("Matrix after zeroing above the diagonal:\n", matrix2_mod)

print("\n=== N3 ===")

start = int(input("Start of range: "))
end = int(input("End of range: "))

seq = np.arange(start, end + 1)

if len(seq) < 30:
    print("Error! Not enough elements for the matrix 6x5!")
else:
    matrix3 = seq[:30].reshape(6, 5)
    print("Matrix:\n", matrix3)

    row_sums = np.sum(matrix3, axis=1)
    print("Row totals:", row_sums)

    col_max = np.max(matrix3, axis=0)
    print("Column max:", col_max)


print("\n=== N4 ===")

low = int(input("Lower limit: "))
high = int(input("Upper limit: "))

arr4 = np.random.randint(low, high + 1, 15)
print("Array:", arr4)

negatives = arr4[arr4 < 0]
print("Negative elements:", negatives)

arr4_mod = arr4.copy()
arr4_mod[arr4_mod < 0] = 0
print("Modified array:", arr4_mod)

zero_count = np.sum(arr4_mod == 0)
print("Count of zeros:", zero_count)

print("\n=== N5 ===")

n = int(input("Array length: "))

arr5_1 = np.random.randint(0, 11, n)
arr5_2 = np.random.randint(10, 21, n)

print("First array:", arr5_1)
print("Second array:", arr5_2)

combined = np.concatenate((arr5_1, arr5_2))
print("Combined array:", combined)

sum_arr = arr5_1 + arr5_2
diff_arr = arr5_1 - arr5_2

print("Sum:", sum_arr)
print("Diff:", diff_arr)

print("\n=== N6 ===")

rows = int(input("Count of rows: "))
cols = int(input("Count of columns: "))

matrix6 = np.arange(rows * cols).reshape(rows, cols)
print("Matrix:\n", matrix6)

new_rows = int(input("New rows: "))
new_cols = int(input("New columns: "))

matrix6_new = matrix6.reshape(new_rows, new_cols)
print("Reshaped matrix:\n", matrix6_new)

row_min = np.min(matrix6_new, axis=1)
row_max = np.max(matrix6_new, axis=1)

print("Min by row:", row_min)
print("Max by row:", row_max)

total_sum = np.sum(matrix6_new)
print("Total sum:", total_sum)