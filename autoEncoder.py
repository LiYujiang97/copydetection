import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# import tensorflow as tf
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import gc
import os
from time import time
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import math


def B_ELM(TrainingData_File, TestingData_File, Elm_Type, NumberofHiddenNeurons, ActivationFunction, k):
    # .............Spliting Data for Training & Testing...........
    T = TrainingData_File[:, 0].T
    P = TrainingData_File[:, 1:].T
    Test_t = TestingData_File[:, 0].T
    Test_p = TestingData_File[:, 1:].T
    num_of_train_data = P.shape[1]
    num_of_test_data = Test_p.shape[1]
    num_of_input_neuron = P.shape[0]
    num_of_hidden_neuron = NumberofHiddenNeurons
    print("num_of_train_data = ", num_of_train_data, "\nnum_of_test_data = ", num_of_test_data,
          "\nnum_of_input_neuron = ", num_of_input_neuron)
    # print(k)
    # Random generate input weights InputWeight (w_i) and biases BiasofHiddenNeurons (b_i) of hidden neurons
    input_weight = np.random.uniform(size=(num_of_hidden_neuron, num_of_input_neuron)) * 2 - 1
    bias_weight = np.random.uniform(size=(num_of_hidden_neuron, 1))
    ind = np.ones((1, num_of_train_data))
    bias_matrix = bias_weight * ind
    bias_neuron = np.empty(0)
    bias_neuron = np.append([[bias_neuron]], [[bias_weight]])
    tempH = input_weight.dot(P)
    tempH = tempH + bias_matrix
    print("T.shape = ", T.shape, "\nP.shape = ", P.shape, "\ninput_weight.shape = ", input_weight.shape,
          "\nbias_weight.shape = ", bias_weight.shape, "\nbias_matrix.shape = ", bias_matrix.shape,
          "\nbias_neuron.shape = ", bias_neuron.shape, "\ntempH.shape = ",
          tempH.shape)
    start_train_time = time()
    # B-ELM when number of hidden nodes L=2n-1
    training_accuracy = 0
    scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    outputWeight = 0
    outputWeight1 = 0
    yym = 0
    for i in range(k):
        input_weight = np.random.uniform(size=(num_of_hidden_neuron, num_of_input_neuron)) * 2 - 1
        bias_neuron = np.empty(0)
        bias_weight = np.random.uniform(size=(num_of_hidden_neuron, 1))
        bias_neuron = np.append([[bias_neuron]], [[bias_weight]])
        ind = np.ones((1, num_of_train_data))
        bias_matrix = bias_weight * ind
        tempH = input_weight.dot(P)
        yym = tempH.dot(np.linalg.pinv(P))
        yjx = P.T.dot(yym.T)
        tempH = tempH + bias_matrix
        # print(tempH.shape)
        # Calculate hidden neuron output matrix H
        if (ActivationFunction == "sig"):
            H = 1 / (1 + np.exp(-tempH))
        else:
            H = math.sin(tempH)
        # Calculate output weights OutputWeight (beta_i)
        # print(H.shape)
        pinv = np.linalg.pinv(H.T)
        outputWeight = pinv.dot(T.T)
        Y = H.T * outputWeight
        m = np.ones((1, 1))
        outputWeights = outputWeight * m

        # B-ELM when number of hidden nodes L=2n
        FY = []
        if (i == 0):
            FY = Y
        else:
            FY = FY + Y
        E1 = T - Y.T
        d = np.ones((1, k))
        y_true = (T * m).T.dot(m)
        # print(y_true.shape)
        y_pred = Y
        mse = mean_squared_error(y_true, y_pred)
        # Calculate training_accuracy
        training_accuracy = np.sqrt(mse)

        Y2 = np.linalg.pinv(outputWeights.T).dot(E1)
        Y2 = Y2.reshape(-1, 1)
        scaler.fit(Y2)
        Y22 = scaler.transform(Y2)
        Y222 = Y2
        Y2 = Y22.T
        T1 = outputWeights.T.dot(Y2)
        if (ActivationFunction == "sig"):
            Y3 = 1 / Y2
            Y3 = Y3 - 1
            Y3 = np.log(Y3)
            Y3 = -Y3
        else:
            Y3 = math.asin(Y2)
        T2 = outputWeights.T.dot(Y3)

        Y4 = Y3
        yym = Y4.dot(np.linalg.pinv(P))
        yjx = P.T.dot(yym.T)

        GXZ111 = P.T.dot(yym.T)
        if ActivationFunction == "sig":
            GXZ2 = 1 / (1 + np.exp(-GXZ111))
        else:
            GXZ2 = math.asin(GXZ111)
        FYY = scaler.inverse_transform(GXZ2)
        # print(GXZ2.shape)
        outputWeight1 = np.linalg.pinv(FYY).dot(E1.T)
        FT1 = FYY.dot(outputWeight1)
        FY = FY + FT1
        y_train = FT1.T
        y_pred = E1
        # print(y_train.shape)
        mse = mean_squared_error(y_train, y_pred)
        # Calculate training_accuracy
        training_accuracy = np.sqrt(mse)
        # input_weight = yym
    end_time = time()
    time_taken_training = end_time - start_train_time
    # print("training time")
    # print(time_taken_training)

    ######################Test############################
    start_test_time = time()
    m = np.ones((1, 1))
    tempH_test = input_weight.dot(Test_p)
    # print(tempH_test.shape)
    ind = np.ones((1, num_of_test_data))
    # print(ind.shape)
    e = np.ones((1, 1))
    bias_matrix = (bias_neuron * e).T.dot(ind)
    # print(bias_matrix.shape)
    tempH_test = tempH_test + bias_matrix
    if (ActivationFunction == "sig"):
        H_test = 1 / (1 + np.exp(-tempH_test))
    else:
        H_test = math.asin(tempH_test)
    # print(H_test.shape)
    TY1 = H_test.T.dot(outputWeight)
    TY = TY1
    E1 = Test_t.T - TY1
    # print(E1.shape)
    testing_accuracy = 0
    for i in range(k):
        GXZ1 = yym.dot(Test_p)
        # print(GXZ1.shape)
        if (ActivationFunction == "sig"):
            GXZ2 = 1 / (1 + np.exp(-GXZ1))
        else:
            GXZ2 = math.asin(GXZ1)
        # print(GXZ2.shape)
        # GXZ2 = GXZ2*m
        FYY = scaler.inverse_transform(GXZ2)
        # print(FYY.shape)
        # D_Beta1 = D_Beta1 * m
        TY2 = FYY.T.dot(outputWeight1)
        # print(TY2.shape)
        y_true = TY2
        # print(y_true.shape)
        # (T * m).T.dot(d)
        y_pred = (E1 * m).T.dot(m)
        # print(y_pred.shape)
        mse = mean_squared_error(y_true, y_pred)
        # print(mse.shape)
        testing_accuracy = np.sqrt(mse)
    # print("testAccuracy")
    # print(testing_accuracy)
    end_test_time = time()
    testing_time = end_test_time - start_test_time

    return (time_taken_training, testing_time, training_accuracy, testing_accuracy)


# # __name__是Python中一个隐含的变量它代表了模块的名字
# # 只有被Python解释器直接执行的模块的名字才是__main__
# if __name__ == '__main__':
def run():
    for k in range(20):
        k = k + 1
        # ...........Load Data..........................
        data = pd.read_csv("cvs/olivettifaces.cvs")
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(data)
        num_array = scaler.transform(data)
        num_array = num_array.T
        num_array[10:] = (num_array[10:]) / 2 + 0.5
        np.random.shuffle(num_array)
        num_array = num_array.T
        training = num_array[0:20768, :]
        test = num_array[20768:, :]

        (TrainingTime, test_time, TrainingAccuracy, TestingAccuracy) = B_ELM(training, test, 0, 1, "sig", k)
        print(TrainingTime)
        print(test_time)
        print(TrainingAccuracy)
        print(TestingAccuracy)
        print("")
