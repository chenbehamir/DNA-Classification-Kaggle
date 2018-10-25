
from __future__ import division, print_function
import numpy as np
import cvxopt #used here with python 3.4

# Hide cvxopt output
cvxopt.solvers.options['show_progress'] = False

class SupportVectorMachine(object):
    """The Support Vector Machine classifier.
    Uses cvxopt to solve the quadratic optimization problem.

    Parameters:
    -----------
    C: float
        Penalty term.
    kernel: function
    """
    def __init__(self, C=1):
        self.C = C
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None
        self.ind= None

    def fit(self, X, y):
        n_samples, n_features = np.shape(X)
        yn = y.values.astype(np.double)
        yn[yn == 0] = -1.0
        
        # Define the quadratic optimization problem 
        P = cvxopt.matrix(np.outer(yn, yn) * X, tc='d')
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        
        A = cvxopt.matrix(yn, (1, n_samples),tc='d')
        b = cvxopt.matrix(0.0, tc='d')

        if not self.C:
            G = cvxopt.matrix(np.identity(n_samples) * -1)
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = np.identity(n_samples) * -1
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        lagr_mult = np.ravel(minimization['x'])

        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        idx = lagr_mult > 1e-5
        self.ind = np.arange(len(lagr_mult))[idx]

        # Get the corresponding lagr. multipliers
        self.lagr_multipliers = lagr_mult[idx]
        
        # Get the samples that will act as support vectors        
        self.support_vector_labels = yn[idx]
        
        print("%d support vectors out of %d points" % (len(self.lagr_multipliers), n_samples))
        # Calculate intercept with first support vector
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept +=  self.support_vector_labels[i]
            self.intercept -= np.sum(self.lagr_multipliers*self.support_vector_labels*X[self.ind[i],idx])
            self.intercept /= len(self.lagr_multipliers)                                 
        
        
    def predict(self, X):
        y_pred = []
        # Iterate through list of samples and make predictions
        for j in range(len(X)):
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                    i] * X[j,self.ind[i]]
            prediction += self.intercept
            y_pred.append(np.sign(prediction))
        return np.array(y_pred)
