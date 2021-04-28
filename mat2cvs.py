import pandas as pd
import scipy
from scipy import io
import os

# matPath = '.\\mat\\'
# outPath = '.\\csv\\'
#
# for i in os.listdir(matPath):
#     inputFile = os.path.join(matPath,"olivettifaces")
#     outputFile = os.path.join(outPath, os.path.split(i)[1][:-4] + '.csv')
#     features_struct = scipy.io.loadmat("mat//olivettifaces.mat")
#     data = list(features_struct.values())[-1]
#     dfdata = pd.DataFrame(data)
#     dfdata.to_csv(outputFile, index=False)
features_struct = scipy.io.loadmat('mat//olivettifaces.mat')
features = list(features_struct.values())[-1]
dfdata = pd.DataFrame(features)
datapath1 = 'E:/workspacelxr/contem/data.txt'
dfdata.to_csv("cvs/da.cvs", index=False)