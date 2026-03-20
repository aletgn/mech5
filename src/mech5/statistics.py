from typing import Tuple

from scipy.optimize import root_scalar
from scipy.special import psi

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


class ExtremeValueStatistic:

    def __init__(self, x: np.ndarray) -> None:
        """
        Class for fitting and analysing extreme values using Gumbel (EV Type-I) distribution.

        Parameters
        ----------
        x : np.ndarray
            1D array of observed extreme values.

        Attributes
        ----------
        x : np.ndarray
            Sorted input data.
        N : int
            Number of observations.
        F_exp : np.ndarray
            Empirical cumulative probabilities.
        ln_F : np.ndarray
            Transformed empirical probabilities for Gumbel plotting.
        mean : float
            Mean of the data.
        std : float
            Standard deviation of the data.
        sigma_mom : float
            Scale parameter estimated from moments.
        mu_mom : float
            Location parameter estimated from moments.
        x_mom : np.ndarray
            Gumbel transformed data using moment estimators.
        sigma_ml : float or None
            Scale parameter estimated by maximum likelihood (None if not fit).
        mu_ml : float or None
            Location parameter estimated by maximum likelihood (None if not fit).
        conf : float or None
            Confidence level used in predict (None if not set).
        var_mu, var_sigma, cov_mu_sigma : float or None
            Variance/covariance estimates for confidence intervals.
        std_mu, std_sigma : float or None
            Standard deviations for confidence intervals.
        """
        self.x = np.sort(x)
        self.N = x.shape[0]

        # CDF
        # self.F_exp = np.arange(1, self.N + 1) / (self.N + 1)
        # self.ln_F = -np.log(-np.log(self.F_exp))
        self.F_exp = self.obs(self.N)
        self.ln_F = self.cdf(self.F_exp)

        # Moments
        self.mean = np.mean(self.x)
        self.std = np.std(self.x, ddof=1)

        self.sigma_mom = (self.std * np.sqrt(6)) / np.pi
        self.mu_mom = self.mean - 0.5772 * self.sigma_mom
        self.x_mom = self.mu_mom + self.sigma_mom * self.ln_F

        # Solution
        self.mu_ml = None
        self.sigma_ml = None

        # Confidence interval - parameters
        self.conf = None
        self.var_mu = None
        self.std_mu = None
        self.var_sigma = None
        self.std_sigma = None
        self.cov_mu_sigma = None

        self.name = "Parameters unavailable"


    @staticmethod
    def obs(size):
        return np.arange(1, size + 1) / (size + 1)


    @staticmethod
    def cdf(obs):
        """Cumulative Distribution Function for Gumbel"""
        return -np.log(-np.log(obs))


    def likelihood(self, sigma_ml: float, mean: float) -> float:
        """
        Likelihood equation for maximum likelihood estimation (scale parameter).

        Parameters
        ----------
        sigma_ml : float
            Current guess for the scale parameter.
        mean : float
            Mean of the input data.

        Returns
        -------
        float
            Value of the likelihood function.
        """
        num = np.sum(self.x * np.exp(-self.x / sigma_ml))
        denom = np.sum(np.exp(-self.x / sigma_ml))
        return mean - num / denom - sigma_ml


    def confidence(self) -> None:
        """
        Compute variance and covariance estimates for the maximum likelihood
        parameters, required for confidence intervals.
        """

        gamma = -psi(1) # Euler Mascheroni constant
        if self.sigma_ml is None:
            raise ValueError("Must run MLE")

        self.var_mu = ((self.sigma_ml**2) / self.N) * (1 + (6 / (np.pi**2)) * ((1-gamma)**2))
        self.std_mu = np.sqrt(self.var_mu)
        self.var_sigma = ((self.sigma_ml**2) / self.N) * (6 / (np.pi**2))
        self.std_sigma = np.sqrt(self.var_sigma)
        self.cov_mu_sigma = ((self.sigma_ml**2) / self.N) * ((6 / (np.pi**2)) * (1-gamma))


    def fit(self, x0: float=None, bracket: Tuple[np.ndarray]=None, method: str="brentq") -> Tuple[float]:
        """
        Fit the Gumbel distribution using maximum likelihood estimation.

        Parameters
        ----------
        x0 : float, optional
            Initial guess for the scale parameter. Defaults to moment estimator.
        bracket : list of np.ndarray, optional
            Bracket for root finding if method requires it.
        method : str
            Method to use in root_scalar. Defaults to "brentq".

        Returns
        -------
        list
            [mu_ml, sigma_ml], fitted location and scale parameters.
        """
        x0 = x0 if x0 is not None else self.sigma_mom

        sol = root_scalar(self.likelihood,
                            args=(self.mean,), x0=x0,
                            method=method, bracket=bracket)
        self.sigma_ml = sol.root
        self.mu_ml = - self.sigma_ml* np.log((np.sum(np.exp(-self.x / self.sigma_ml) ) / self.N ) );

        print(f"MLE estimate: {self.sigma_ml}")
        self.confidence()
        self.name = fr"$y = {self.mu_ml:.2f} + {self.sigma_ml:.2f}(-\ln(-\ln(F)))$"
        return self.mu_ml, self.sigma_ml


    def predict(self, x_edges: np.ndarray=None, conf:float=0.95) -> Tuple[np.ndarray]:
        """
        Predict the Gumbel distribution and confidence intervals over a domain.

        Parameters
        ----------
        x_edges : np.ndarray, optional
            Two-element array specifying [xmin, xmax] for extended prediction.
            If None, uses the original data points.
        conf : float
            Confidence level for interval (default 0.95).

        Returns
        -------
        x_ml : np.ndarray
            Predicted Gumbel values.
        ln_F : np.ndarray
            Transformed probabilities used in prediction.
        x_ml_lo : np.ndarray
            Lower confidence interval.
        x_ml_up : np.ndarray
            Upper confidence interval.
        """
        self.conf = conf

        if self.sigma_ml is None:
            raise ValueError("Run MLE first.")
        if self.var_mu is None:
            raise ValueError("Must run confidence first.")


        if x_edges is None:
            x_pred = self.x
            ln_F = self.ln_F

        else:
            x_pred = np.arange(x_edges[0], x_edges[1], 1)
            print(x_pred)
            N = x_pred.shape[0]
            F_exp = self.obs(N)
            ln_F = self.cdf(F_exp)
            # F_exp = np.arange(1, N + 1) / (N + 1)
            # ln_F = -np.log(-np.log(F_exp))

        x_ml = self.mu_ml + self.sigma_ml * ln_F
        var_F = self.var_mu + 2 * self.cov_mu_sigma * ln_F + self.var_sigma * ln_F**2

        z = norm.ppf((1+self.conf)/2)
        x_ml_lo = x_ml - z * np.sqrt(var_F)
        x_ml_up = x_ml + z * np.sqrt(var_F)

        return x_ml, ln_F, x_ml_lo, x_ml_up


    def inspect(self, x_edges: Tuple[float]=None) -> None:
        """
        Plot the empirical data and the fitted Gumbel distribution with confidence intervals.

        Parameters
        ----------
        x_edges : list of two floats, optional
            Domain to extend the plot. Defaults to experimental data.
        """
        fig, ax = plt.subplots()

        if self.x is not None:
            ax.scatter(self.x, self.ln_F, color="none", edgecolors="k", label="Data")

        if self.var_mu is not None:
            x, ln_F, low, up = self.predict(x_edges)
            ax.plot(x, ln_F, '-k', zorder=-1,
                    label=fr"$y = {self.mu_ml:.3f} + {self.sigma_ml:.3f}(-\ln(-\ln(F)))$")
            ax.plot(low, ln_F,  '--k', zorder=-1, label=f"Lower @ {self.conf}")
            ax.plot(up, ln_F, '--k', zorder=-1, label=f"Upper @ {self.conf}")

        ax.tick_params("both", direction="in", right=1, top=1)

        plt.xlabel(r'Max', fontsize=12)
        plt.ylabel(r'$-\ln{(-\ln{(F)})}$', fontsize=12)

        plt.legend(loc="best")
        plt.show()


    def __repr__(self):
        return self.name


if __name__ == "__main__":
    # evs = ExtremeValueStatistic(data_1)
    # evs.fit(bracket=[0.1, 50])
    # evs.confidence()
    # # evs.predict()
    # evs.inspect([10, 90])
    ...