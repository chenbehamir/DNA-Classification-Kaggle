# DNA Classification

 This project was of the form of a private Kaggle competition ([here](https://www.kaggle.com/c/advanced-learning-models)). 
 
 
 
This directory contains: <br/>
	- kernelComputation.py : functions for computation of Gram matrices <br/>
	- svmImpl : contains the SVM class implementation (fit and predict functions) <br/>
	- start.py :  contains the final execution to obtain a submission file <br/>
	- start_alternative.py : contains the final execution but here Gram matrices were precomputed and store in files. This has been done because the Gram matrices computation of the three data sets takes one hour <br/>
	- preComput.py : the script that pre-compute Gram matrices <br/>

To start : python start.py or python start_alternative.py
