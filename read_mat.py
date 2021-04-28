import scipy.io as io
import numpy as np

data = io.loadmat('E:\mytest.mat') # data是一个dictionary
print("mytest",data)

data_1 = io.loadmat('E:\olivettifaces.mat')
print("livettifaces=",data_1)