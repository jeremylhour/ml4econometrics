"""
ols:

Created on Sat Jul 23 22:36:34 2023

@author: jeremylhour
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.metrics import r2_score

@dataclass
class OLS(BaseEstimator):
    """
    OLS:
        Ordinary Least Squares,
        but outputs variance, etc.
    """
    fit_intercept: bool = True
    robust_variance: bool = True
    rcond: float = 1e-15

    def fit(self, X, y):
        """
        estimate:
            Compute the OLS coefficient

        Args:
            X (np.array): Dimension (n, p).
            y (np.array): Dimension (n,).
        """
        if self.fit_intercept:
            X = np.c_[np.ones(len(X)), X]

        beta, _, rank_, singular_ = np.linalg.lstsq(X, y, rcond=None)
        if self.fit_intercept:
            self.intercept_ = beta[0]
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.
            self.coef_ = beta
        return self

    def predict(self, X):
        check_is_fitted(self, 'coef_')
        check_is_fitted(self, 'intercept_')
        X = check_array(X)
        return self.intercept_ + X @ self.coef_
    
    def score(self, X, y):
        y_hat = self.predict(X)
        return r2_score(y, y_hat)
    
    def compute_std(self, X, y, robust=True):
        """
        compute_std:
        
        Args:
            X (np.array): Dimension (n, p).
            y (np.array): Dimension (n,).
            robust (bool): If True, uses White sandwich variance estimator. 
        """
        check_is_fitted(self, 'coef_')
        check_is_fitted(self, 'intercept_')
        
        epsilon = y - self.predict(X)

        if self.fit_intercept:
            X = np.c_[np.ones(len(X)), X]

        XtX = X.T @ X
        inv_XtX = np.linalg.pinv(XtX, rcond=self.rcond, hermitian=True)

        if robust:
            A = (inv_XtX @ X.T) * epsilon
            self.variance_ =  len(X) * A @ A.T
        else:
            self.variance_ =  len(X) * np.sum(epsilon ** 2) * inv_XtX / (len(X) - 1)

        self.std = np.sqrt(np.diag(self.variance_) / len(X))

    def compute_student_statistics(self, X, y):
        check_is_fitted(self, 'coef_')

        if not hasattr(self, "variance_"):
            self.compute_std(X=X, y=y, robust=self.robust_variance)

        if self.fit_intercept:
            self.t_stat = self.coef_ / self.std[1:]
        else:
            self.t_stat = self.coef_ / self.std
        self.p_value = 1 - norm.cdf(np.abs(self.t_stat))