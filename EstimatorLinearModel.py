#Stevie Merrill
#captainbillybob23@gmail.com
"""This will contain the estimator that my project will use"""
from sklearn import linear_model
from sklearn.base import BaseEstimator

class Estimator(BaseEstimator):
    """Functional Estimator"""
    ####################API Required
    
    def __init__(self):
        self.model = linear_model.LinearRegression()
    
    def fit(self,X,y):
        self.is_fitted_ = True
        self.model.fit(X,y)
        self.coef = self.model.coef_[0,0]
        self.intercept = self.model.intercept_[0]

    def predict(self,X):
        return self.model.predict(X)
    
    def score(self,X,y):
        return self.model.score(X,y)

    ################## attributes
