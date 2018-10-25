# DNA Classification


This directory contains:
	- kernelComputation.py : functions for computation of Gram matrices
	- svmImpl : contains the SVM class implementation (fit and predict functions)
	- start.py :  contains the final execution to obtain a submission file
	- start_alternative.py : contains the final execution but here Gram matrices were precomputed and store in files. This has been done because the Gram matrices computation of the three data sets takes one hour
	- preComput.py : the script that pre-compute Gram matrices

To start : python start.py or python start_alternative.py
