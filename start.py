import pandas as pd
from kernelComputation import computeKernelMatrix
from svmImpl import SupportVectorMachine
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
print("computing the kernel matrix for the first data set")
print("This may take a while")
kernel_train_0 = computeKernelMatrix(X_train_0,X_train_0, 5)

print("computing the kernel matrix for the second data set")
print("This may take a while")
kernel_train_1 = computeKernelMatrix(X_train_1,X_train_1, 5)

print("computing the kernel matrix for the third data set")
print("This may take a while")
kernel_train_2 = computeKernelMatrix(X_train_2,X_train_2, 5)

#computing testing gram matrix
print("computing the kernel matrix for the first test set")
print("This may take a while")
kernel_test_0 = computeKernelMatrix(X_test_0, X_train_0, 5)

print("computing the kernel matrix for the second test set")
print("This may take a while")
kernel_test_1 = computeKernelMatrix(X_test_1, X_train_1, 5)

print("computing the kernel matrix for the third test set")
print("This may take a while")
kernel_test_2 = computeKernelMatrix(X_test_2, X_train_2, 5)

#run SVM method on first data set
model_SVM_0 = SupportVectorMachine(C= 1e-3)
model_SVM_0.fit(kernel_train_0, y_train_0)
predicted_svm_0 = model_SVM_0.predict(kernel_test_0)
predicted_svm_0[predicted_svm_0 == -1] = 0
predicted_svm_0 = predicted_svm_0.astype(int)


#run SVM method on second data set
model_SVM_1 = SupportVectorMachine(C= 1e1) 
model_SVM_1.fit(kernel_train_1, y_train_1)
predicted_svm_1 = model_SVM_1.predict(kernel_test_1)
predicted_svm_1[predicted_svm_1 == -1] = 0
predicted_svm_1 = predicted_svm_1.astype(int)

#run SVM method on third data set
model_SVM_2 = SupportVectorMachine(C= 1e-3)
model_SVM_2.fit(kernel_train_2, y_train_2)
predicted_svm_2 = model_SVM_2.predict(kernel_test_2)
predicted_svm_2[predicted_svm_2 == -1] = 0
predicted_svm_2 = predicted_svm_2.astype(int)

predstmp_svm = np.concatenate((predicted_svm_0, predicted_svm_1, predicted_svm_2),axis = 0)
print(len(predstmp_svm ))
Ids_svm = [i for i in range(3000)]
d_svm = {'Id': Ids_svm,'Bound': predstmp_svm}
df = pd.DataFrame(d_svm,columns = ['Id','Bound'])
df.to_csv("Yte.csv",index = None)
