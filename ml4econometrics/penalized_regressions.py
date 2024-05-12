"""
penalized regressions:

Created on Sat Jul 23 22:36:34 2023

@author: jeremylhour
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np
from tqdm import tqdm
from scipy.stats import norm

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import r2_score

from ml4econometrics.ols import OLS

#------------------------------------------------------
# LASSO
#------------------------------------------------------
@dataclass
class Lasso(BaseEstimator):
    """
    Lasso:
        Lasso FISTA estimation.

    Args:
        alpha (float): Overall penalty strength.
        nopen (list of int): Index of variables that should not be penalized.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for convergence.
        fit_intercept (bool): If True, fits an intercept.
        verbose (bool): If True, displays progress.

    Attributes
    coef_ : array, shape (n_features,)
        Estimated coefficients.
    """
    alpha: float = 1.
    nopen: List[int] = field(default_factory=list)
    fit_intercept: bool = True
    verbose: bool = False
    max_iter: int = 5_000
    tol: float = 1e-6

    def loss_gradient(self, w, b, y, X):
        return - 2 * (y - b - X @ w) @ X / len(X)
    
    def intercept_loss_gradient(self, w, b, y, X):
        return - 2 * (y - b - X @ w).mean()
        
    def prox(self, x, alpha, nopen):
        y = np.sign(x) * np.maximum(0, np.abs(x) - alpha)
        if nopen is not None:
            y[nopen] = x[nopen]
        return y

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_, n_features = X.shape

        # Initialize constants and variables
        _, w, _ = np.linalg.svd(X)
        eta = n_ / (2 * w[0] ** 2)
        t, t_old = 1., 1.
        coef, z = np.zeros(n_features), np.zeros(n_features)
        b = 0.

        if self.verbose:
            loop = tqdm(range(self.max_iter))
        else:
            loop = range(self.max_iter)

        for _ in loop:
            # Save parameters
            coef_old = coef.copy() 
            t_old = t
            t = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
            delta = (1 - t_old) / t

            # Compute gradient and update coefficients
            grad = self.loss_gradient(z, b, y, X)
            coef = self.prox(
                x=z - eta * grad,
                alpha=eta * self.alpha,
                nopen=self.nopen
                )
            if self.fit_intercept:
                b  -= eta * self.intercept_loss_gradient(w=coef, b=b, y=y, X=X)
            z = (1 - delta) * coef + delta * coef_old

            # Check convergence
            if np.linalg.norm(coef - coef_old) / np.maximum(1e-9, np.linalg.norm(coef)) < self.tol:
                break
        if (_ == self.max_iter - 1):
            print("Max. number of iterations reached.")

        self.coef_ = coef
        self.intercept_ = b
        return self

    def predict(self, X):
        check_is_fitted(self, 'coef_')
        check_is_fitted(self, 'intercept_')
        X = check_array(X)
        return self.intercept_ + X @ self.coef_
    
    def score(self, X, y):
        y_hat = self.predict(X)
        return r2_score(y, y_hat)

#------------------------------------------------------
# ITERATIVE LASSO OF BELLONI, CHERNOZHUKOV AND HANSEN
#------------------------------------------------------
@dataclass
class BCHLasso(BaseEstimator):
    """
    BCHLasso
    """
    fit_intercept: bool = True
    verbose: bool = False
    max_iter: int = 5_000
    tol: float = 1e-6
    max_iter_phi: int = 200
    tol_phi: float = 1e-6
    post_lasso: bool = True

    def loss_gradient(self, w, b, y, X):
        return - 2 * (y - b - X @ w) @ X / len(X)
    
    def intercept_loss_gradient(self, w, b, y, X):
        return - 2 * (y - b - X @ w).mean()
        
    def prox(self, x, alpha, phi):
        y = np.sign(x) * np.maximum(0, np.abs(x) - alpha * phi)
        return y

    def fit_one_step(self, X, y, phi):
        n_, n_features = X.shape

        # Initialize constants and variables
        _, w, _ = np.linalg.svd(X)
        eta = n_ / (2 * w[0] ** 2)
        t, t_old = 1., 1.
        coef, z = np.zeros(n_features), np.zeros(n_features)
        b = 0.

        for _ in range(self.max_iter):
            # Save parameters
            coef_old = coef.copy() 
            t_old = t
            t = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2
            delta = (1 - t_old) / t

            # Compute gradient and update coefficients
            grad = self.loss_gradient(z, b, y, X)
            coef = self.prox(
                x=z - eta * grad,
                alpha=eta * self.alpha,
                phi=phi
                )
            if self.fit_intercept:
                b  -= eta * self.intercept_loss_gradient(w=coef, b=b, y=y, X=X)
            z = (1 - delta) * coef + delta * coef_old

            # Check convergence
            if np.linalg.norm(coef - coef_old) / np.maximum(1e-9, np.linalg.norm(coef)) < self.tol:
                break
        if (_ == self.max_iter - 1):
            print("Max. number of iterations reached.")

        self.coef_ = coef
        self.intercept_ = b
        return self
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        n_, n_features = X.shape

        # BCH Penalty level
        g = .1 / np.log(max(n_, n_features))
        self.alpha =  2 * 1.1 * norm.ppf(1 - .5 * g / n_features) / np.sqrt(n_) # (theoretical) Lasso penalty level

        # Initial guess fot phi
        eps = y - y.mean()
        phi = np.sqrt((eps ** 2).T @ (X ** 2) / len(X))
        phi0 = phi

        if self.verbose:
            loop = tqdm(range(self.max_iter_phi))
        else:
            loop = range(self.max_iter_phi)

        for _ in loop:
            phi0 = phi
            self.fit_one_step(X, y, phi)

            eps = y - self.predict(X)
            phi = np.sqrt((eps ** 2).T @ (X ** 2) / len(X))

            # Check convergence
            if np.linalg.norm(phi - phi0) / np.maximum(1e-9, np.linalg.norm(phi)) < self.tol_phi:
                break
        if (_ == self.max_iter_phi - 1):
            print("Max. number of iterations for iterative penalty estimation reached.")

        #  Post-Lasso step
        if self.post_lasso:
            self.s_ = np.nonzero(self.coef_)[0]
            ols = OLS(fit_intercept=self.fit_intercept)
            ols.fit(X[:, self.s_], y)

            self.coef_ = np.zeros_like(self.coef_)
            self.coef_[self.s_] = ols.coef_
            self.intercept_ = ols.intercept_
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'coef_')
        check_is_fitted(self, 'intercept_')
        X = check_array(X)
        return self.intercept_ + X @ self.coef_
    
    def score(self, X, y):
        y_hat = self.predict(X)
        return r2_score(y, y_hat)

#------------------------------------------------------
# LOGIT LASSO
#------------------------------------------------------
class LogitLasso(Lasso):
    def loss_gradient(self, w, b, y, X):
        return (1 / (1 + np.exp(- b - X @ w)) - y) @ X / len(X)
    
    def intercept_loss_gradient(self, w, b, y, X):
        return (1 / (1 + np.exp(- b - X @ w)) - y).mean()
    
    def predict(self, X):
        check_is_fitted(self, 'coef_')
        check_is_fitted(self, 'intercept_')
        X = check_array(X)
        return 1. / (1. + np.exp(- self.intercept_ - X @ self.coef_))