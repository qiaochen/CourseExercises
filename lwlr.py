from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.base import BaseEstimator
import numpy as np
import sys


class LWLR(BaseEstimator):
    """
    A demo of Local Weight Linear Regression, 
    refer to https://www.geeksforgeeks.org/ml-locally-weighted-linear-regression/
    """
    def __init__(self,k):
        """
        The hyper parameter k controls the weighting effect for samples
        """
        self.k = k
    
    def fit(self,X,y):
        """
        Non-parametric method, memorize training samples X and targets y
        """
        self.X = X
        self.y = y
        return self
        
    def predict(self, X):
        """
        Given feature vectors X, make prediction
        """
        result = []
        for example in X:
            prediction = self._predict_single(example)
            result.append(prediction)
            
        return np.array(result)
        
    def _predict_single(self, X_in):
        """
        Train and predict for a sinlge input feature vector X_in
        """
        # compute weights
        weights = np.exp(-np.sum(np.square(self.X - X_in), axis=1)/(2*self.k))
        
        # fit linear regression model on weighted samples
        lr = LinearRegression().fit(self.X, self.y, sample_weight=weights)
        
        # make prediction
        return lr.predict(X_in.reshape(1,-1))
    
    
def search_best_k(X, y, n_folds=10, k_range=[1], scoring="neg_mean_squared_log_error"):
    """
    Search best k given a list of candidate ks
    n_folds: the number of CV split
    k_range: list of candidate ks
    scoring: sklearn acceptable scoring object or strings for model selection
    Return: tuple of (the best k, cv history)
    """
    np.random.seed(999)
    results = []
    best_k, best_score = 0, -sys.maxsize
    for k in k_range:
        scores = cross_validate(LWLR(k), X, y, cv=10, scoring=scoring, n_jobs=-1, return_train_score=True)
        avg_train_score = np.mean(scores['train_score'])
        avg_test_score = np.mean(scores["test_score"])
        results.append({"k":k, "avg_train_score":avg_train_score, "avg_test_score":avg_test_score})
        if best_score < avg_test_score:
            best_k, best_score = k, avg_test_score
        print(results[-1])
    return best_k, results
