import numpy as np


x = np.ones(shape=(2, 2))
a = 1
def func(param):
    param += 1

func(x)
print(x)
func(a)
print(a)
