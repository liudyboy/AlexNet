import numpy as np
def quantization(array):
    Max = np.amax(array)
    Min = np.amin(array)
    denominator = Max - Min
    numerator = (pow(2., 8.)-1.) * (array - Min)
    quan_array = numerator/denominator
    quan_array = np.array(quan_array, dtype=np.uint8)
    return quan_array

if __name__ == '__main__':
    array = np.arange(12).reshape(3,4)
    array = np.array(array, dtype=np.float32)
    array = quantization(array)
    print(array)
