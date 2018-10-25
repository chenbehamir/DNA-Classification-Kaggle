import pandas as pd
from kernelComputation import computeKernelMatrix
import numpy as np

#load data

#Load training samples
X_train_0 = pd.read_csv("data/Xtr0.csv", header= None, delim_whitespace= True).as_matrix(columns=None)
X_train_1 = pd.read_csv("data/Xtr1.csv", header= None, delim_whitespace= True).as_matrix(columns=None)
X_train_2 = pd.read_csv("data/Xtr2.csv", header= None, delim_whitespace= True).as_matrix(columns=None)

#load training labels

y_train_0 = pd.read_csv("data/Ytr0.csv")["Bound"]
y_train_1 = pd.read_csv("data/Ytr1.csv")["Bound"]
y_train_2 = pd.read_csv("data/Ytr2.csv")["Bound"]

#testing samples
X_test_0 = pd.read_csv("data/Xte0.csv", header= None, delim_whitespace= True).as_matrix(columns=None)
X_test_1 = pd.read_csv("data/Xte1.csv", header= None, delim_whitespace= True).as_matrix(columns=None)
X_test_2 = pd.read_csv("data/Xte2.csv", header= None, delim_whitespace= True).as_matrix(columns=None)

#computing training gram matrix
kernel_train_0 = computeKernelMatrix(X_train_0,X_train_0, 5)
kernel_train_1 = computeKernelMatrix(X_train_1,X_train_1, 5)
kernel_train_2 = computeKernelMatrix(X_train_2,X_train_2, 5)


#computing testing gram matrix
kernel_test_0 = computeKernelMatrix(X_test_0, X_train_0, 5)
kernel_test_1 = computeKernelMatrix(X_test_1, X_train_1, 5)
kernel_test_2 = computeKernelMatrix(X_test_2, X_train_2, 5)


#save the matrixes somewhere to avoid recomputation
df0 = pd.DataFrame(data=kernel_train_0)
df0.to_csv('kernelMatrixes/Kernek_train_0.csv', index=None)

df1 = pd.DataFrame(data=kernel_train_1)
df1.to_csv('kernelMatrixes/Kernek_train_1.csv', index=None)

df2 = pd.DataFrame(data=kernel_train_2)
df2.to_csv('kernelMatrixes/Kernek_train_2.csv', index=None)


dfT0 = pd.DataFrame(data=kernel_test_0)
dfT0.to_csv('kernelMatrixes/Kernek_test_0.csv', index=None)


dfT1 = pd.DataFrame(data=kernel_test_1)
dfT1.to_csv('kernelMatrixes/Kernek_test_1.csv', index=None)

dfT2 = pd.DataFrame(data=kernel_test_2)
dfT2.to_csv('kernelMatrixes/Kernek_test_2.csv', index=None)