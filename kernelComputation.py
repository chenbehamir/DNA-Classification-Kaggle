import numpy as np
#spectrum kernel implem
#saw it in ALM course


def getSubString(mString, spectrum):
    """
    get substrings of length spectrum
    
    Attributes:
        mstring: the string to subdivise
        spectrum : the length of the substrings
    return :
        substrings of length spectrum
    """
    tmpList = []
    mString = mString.lower()
    if (spectrum == 0):
        tmpList = ['']
    else:
        for i in range(len(mString)-spectrum+1):
            mStringRes = ''
            for j in range(spectrum):
                mStringRes += mString[i+j]
            tmpList.append(mStringRes)
    return tmpList

def pSpectrumKernelFunction(mString1, mString2, spectrum):
    """
    compute the spectrum kernel of two strings
   
    Attributes:
        mstring1 : the first string
        mstring2: the second string
        spectrum : the length of substrings(spectrum used)
    return :
        spectrum kernel value
    """
    
    subString1 = getSubString(mString1, spectrum)
    subString2 = getSubString(mString2, spectrum)
    kernel = 0
    for i in subString1:
        for j in subString2:
            if (i==j):
                kernel += 1
    return kernel

def _gram_matrix_element(mString1, mString2, spectrum, sdkvalue1, sdkvalue2):
    """
        compute the element K(i,j) of a gram matrix
        normalize spectrum kernel are used Knorm(i,j) = K(i,j)/(K(i,i)*K(j,j))^O.5
        
        Attributes:
            mstring1, mstring2: strings for which we compute the kernel
            sdkvalue1, sdvalue2: K(mString1,mString1) and K(mString2,mString2) are diagonal 
                                  elements of K
    """
    if mString1 == mString2:
        return 1
    else:
        try:
            return pSpectrumKernelFunction(mString1, mString2, spectrum) / \
                       (sdkvalue1 * sdkvalue2) ** 0.5
        except ZeroDivisionError:
            print("Maximal subsequence length is less or equal to documents' minimal length."
                      "You should decrease it")
            sys.exit(2)
            
def computeKernelMatrix(X1, X2, spectrum):
    """
    spectrum Kernel computation
    Attributes:
        param X: list of DNAs (m rows, 1 column); each row is a single DNA (string)
        return: Gram matrix for the given parameters
    """
    len_X1 = len(X1)
    len_X2 = len(X2)
    
    # numpy array of Gram matrix
    gram_matrix = np.zeros((len_X1, len_X2), dtype=np.float32)
    sim_docs_kernel_value = {}
    #when lists of documents are identical
    if np.array_equal(X1,X2):#.all():
    #store K(s,s) values in dictionary to avoid recalculations
        for i in range(len_X1):
#             print("ii",i)
            sim_docs_kernel_value[i] = pSpectrumKernelFunction(X1.item(i), X1.item(i), spectrum)
        #calculate Gram matrix
        for i in range(len_X1):
#             print("i",i)
            for j in range(i, len_X2):
                if(i==j):
                    gram_matrix[i, j] = 1.0
                else:
                    gram_matrix[i, j] = _gram_matrix_element(X1.item(i), X2.item(j), spectrum, sim_docs_kernel_value[i]
                                                                 ,sim_docs_kernel_value[j])
            #using symmetry
                    gram_matrix[j, i] = gram_matrix[i, j]
        
    #when lists of documents are not identical but of the same length
    elif len_X1 == len_X2:
        sim_docs_kernel_value[1] = {}
        sim_docs_kernel_value[2] = {}
        #store K(s,s) values in dictionary to avoid recalculations
        for i in range(len_X1):
            sim_docs_kernel_value[1][i] = pSpectrumKernelFunction(X1.item(i), X1.item(i), spectrum)
        for i in range(len_X2):
            sim_docs_kernel_value[2][i] = pSpectrumKernelFunction(X2.item(i), X2.item(i), spectrum)
        #calculate Gram matrix
        for i in range(len_X1):
#             print("ilen",i)
            for j in range(i, len_X2):
                gram_matrix[i, j] = _gram_matrix_element(X1.item(i), X2.item(j), spectrum, sim_docs_kernel_value[1][i],
                                                             sim_docs_kernel_value[2][j])
        #using symmetry
                gram_matrix[j, i] = gram_matrix[i, j]
    
    #when lists of documents are neither identical nor of the same length
    else:
        sim_docs_kernel_value[1] = {}
        sim_docs_kernel_value[2] = {}
        min_dimens = min(len_X1, len_X2)
        #store K(s,s) values in dictionary to avoid recalculations
        for i in range(len_X1):
            sim_docs_kernel_value[1][i] = pSpectrumKernelFunction(X1.item(i), X1.item(i), spectrum)
        for i in range(len_X2):
            sim_docs_kernel_value[2][i] = pSpectrumKernelFunction(X2.item(i), X2.item(i), spectrum)
        #calculate Gram matrix for square part of rectangle matrix
        for i in range(min_dimens):
#             print("ielse1",i)
            for j in range(i, min_dimens):
                gram_matrix[i, j] = _gram_matrix_element(X1.item(i), X2.item(j), spectrum, sim_docs_kernel_value[1][i],
                                                             sim_docs_kernel_value[2][j])
                    #using symmetry
                gram_matrix[j, i] = gram_matrix[i, j]

        #if more rows than columns
        if len_X1 > len_X2:
            for i in range(min_dimens, len_X1):
                for j in range(len_X2):
                    gram_matrix[i, j] = _gram_matrix_element(X1.item(i), X2.item(j), sim_docs_kernel_value[1][i],
                                                                 sim_docs_kernel_value[2][j])
        #if more columns than rows
            
        else:
            for i in range(len_X1):
#                 print("ielse2",i)
                for j in range(min_dimens, len_X2):
                    gram_matrix[i, j] = _gram_matrix_element(X1.item(i), X2.item(j),spectrum, sim_docs_kernel_value[1][i],
                                                                 sim_docs_kernel_value[2][j])
    return gram_matrix