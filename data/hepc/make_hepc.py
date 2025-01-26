import uci_datasets as uci
import numpy as np

data = uci.Dataset("houseelectric")
x_train, y_train, x_test, y_test = data.get_split(split=0)

np.savez('train_data.npz', x_train=x_train, y_train=y_train)
np.savez('test_data.npz', x_test=x_test, y_test=y_test)