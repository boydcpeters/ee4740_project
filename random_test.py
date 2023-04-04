import cs_func
import process_data


m = 2
n = 3

# This is your code
A = cs_func.create_A(m, n, seed=1)

print(A)

# This is my code
A2 = cs_func.create_A(m, n, seed=1)

print(A2)

print()
print()
mse = process_data.load_arr("data\\biht\\A25_seed1\\mse.npy")
print(mse)