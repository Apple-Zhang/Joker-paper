import numpy as np
import pandas as pd
# Load the CSV file using pandas
data = pd.read_csv('supersymmetry_dataset.csv', header=None)
print(data)

# Convert the data into a numpy array
np_array = data.to_numpy()
print(np_array)
print(np_array.shape)

n_train = 4000000

y_train = np_array[:n_train,0]
y_test  = np_array[n_train:,0]
x_train = np_array[:n_train,1:]
x_test  = np_array[n_train:,1:]

# Save the data into a .npz file
np.savez('susy_train.npz', X=x_train, Y=y_train)
np.savez('susy_test.npz',  X=x_test, Y=y_test)

print(np_array.mean(axis=1), np_array.std(axis=1))

print("npz saved.")
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)