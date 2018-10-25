import pandas as pd
from svmImpl import SupportVectorMachine
import numpy as np


#here the kernel matrices have been already computed and store in files
#This is for speeding up execution since kernel computation take one hout
#Load training kernels
kernel_train_0 = pd.read_csv("kernelMatrixes/Kernek_train_0.csv", header= None,skiprows=1, sep=',').values
kernel_train_1 = pd.read_csv("kernelMatrixes/Kernek_train_1.csv", header= None,skiprows=1, sep=',').values
kernel_train_2 = pd.read_csv("kernelMatrixes/Kernek_train_2.csv", header= None,skiprows=1, sep=',').values


#Load testing kernels
kernel_test_0 = pd.read_csv("kernelMatrixes/Kernek_test_0.csv", header= None,skiprows=1, sep=',').values
kernel_test_1 = pd.read_csv("kernelMatrixes/Kernek_test_1.csv", header= None,skiprows=1, sep=',').values
kernel_test_2 = pd.read_csv("kernelMatrixes/Kernek_test_2.csv", header= None,skiprows=1, sep=',').values



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
