from __future__ import division
import numpy as np
from sys import stdout
from sklearn.metrics import pairwise_kernels
import scipy.stats as st
from sklearn.metrics import pairwise_distances


def treatment_kernel(T1, T2=None, metric="rbf", lam=0.0, gamma=None):
    """
    Selects the appropriate kernel:
    - Discrete kernel if T1 is binary
    - RBF kernel otherwise
    """
    if T2 is None:
        T2 = T1

    is_binary = np.unique(T1).size <= 2 and np.all(np.isin(T1, [0, 1]))

    if is_binary:
        # Discrete kernel
        K = (T1[:, None] == T2[None, :]).astype(float)
        if T1 is not T2:
            K[K == 0] = lam
        return K
    else:
        # RBF or user-defined kernel
        if gamma is None:
            try:
                dists = (
                    pairwise_distances(T1[:, None], T2[:, None])
                    if T1.ndim == 1
                    else pairwise_distances(T1, T2)
                )
                gamma = 1.0 / (np.median(dists) ** 2 + 1e-8)
            except:
                gamma = 1.0
        return pairwise_kernels(T1[:, None], T2[:, None], metric=metric, gamma=gamma)


def xMMD2dr(
    Y,
    X,
    logging_T,
    w_pi,
    w_pi_prime,
    pi_samples,
    pi_prime_samples,
    kernel_function,
    reg_lambda=1e-2,
    **kwargs
):
    """The DR-xKPE^2 statistic."""
    N = len(Y)

    # sigmaKX = np.median(pairwise_distances(X[N2:, :], X[N2:, :], metric='euclidean'))**2
    sigmaKX = np.median(pairwise_distances(X[:, :], X[:, :], metric="euclidean")) ** 2

    KX = pairwise_kernels(X, metric="rbf", gamma=1.0 / sigmaKX)
    KT = treatment_kernel(logging_T, metric="rbf", lam=0.5)
    KT_pi = treatment_kernel(logging_T, pi_samples, metric="rbf", lam=0.5)
    KT_pi_prime = treatment_kernel(logging_T, pi_prime_samples, metric="rbf", lam=0.5)
    gamma = reg_lambda

    mu_logging = np.linalg.solve(
        np.multiply(KX, KT) + gamma * np.eye(N), np.multiply(KX, KT)
    )
    mu_pi = np.linalg.solve(
        np.multiply(KX, KT) + gamma * np.eye(N), np.multiply(KX, KT_pi)
    )
    mu_pi_prime = np.linalg.solve(
        np.multiply(KX, KT) + gamma * np.eye(N), np.multiply(KX, KT_pi_prime)
    )

    # DR
    dr_term = mu_pi_prime - mu_pi + (w_pi_prime - w_pi) * (np.eye(N) - mu_logging)

    KY = pairwise_kernels(Y, Y, metric=kernel_function, **kwargs)
    prod = dr_term.T @ KY @ dr_term

    U = prod.mean(1)
    return np.sqrt(len(U)) * U.mean() / U.std()


def xMMD2dr_cross_fit(
    Y,
    X,
    logging_T,
    w_pi,
    w_pi_prime,
    pi_samples,
    pi_prime_samples,
    kernel_function,
    reg_lambda=1e-1,
    **kwargs
):
    """The DR-xKPE^2 statistic with cross-fitting"""
    """The DR-xKTE^2 statistic.
    """

    N = len(Y)
    N2 = N // 2

    w_pi_split1 = w_pi[:N2]
    w_pi_split2 = w_pi[N2:]

    w_pi_prime_split1 = w_pi_prime[:N2]
    w_pi_prime_split2 = w_pi_prime[N2:]

    Y_split1 = Y[:N2]
    Y_split2 = Y[N2:]

    # sigmaKX = np.median(pairwise_distances(X[N2:, :], X[N2:, :], metric='euclidean'))**2
    sigmaKX = (
        np.median(pairwise_distances(X[N2:, :], X[N2:, :], metric="euclidean")) ** 2
    )

    sigmaKT = (
        np.median(
            pairwise_distances(
                logging_T[N2:, np.newaxis],
                logging_T[N2:, np.newaxis],
                metric="euclidean",
            )
        )
        ** 2
    )
    KX = pairwise_kernels(X, metric="rbf", gamma=1.0 / sigmaKX)
    # KT = pairwise_kernels(logging_T[:, np.newaxis], metric="rbf", gamma=1.0 / sigmaKT)
    # KT_pi = pairwise_kernels(
    #     logging_T[:, np.newaxis],
    #     pi_samples[:, np.newaxis],
    #     metric="rbf",
    #     gamma=1.0 / sigmaKT,
    # )
    # KT_pi_prime = pairwise_kernels(
    #     logging_T[:, np.newaxis],
    #     pi_prime_samples[:, np.newaxis],
    #     metric="rbf",
    #     gamma=1.0 / sigmaKT,
    # )
    KT = treatment_kernel(logging_T, metric="rbf")
    KT_pi = treatment_kernel(logging_T, pi_samples, metric="rbf")
    KT_pi_prime = treatment_kernel(logging_T, pi_prime_samples, metric="rbf")
    gamma = reg_lambda

    mu_logging_split1 = np.linalg.solve(
        np.multiply(KX[:N2, :N2], KT[:N2, :N2]) + gamma * np.eye(N2),
        np.multiply(KX[:N2, :N2], KT[:N2, :N2]),
    )
    mu_pi_split1 = np.linalg.solve(
        np.multiply(KX[:N2, :N2], KT[:N2, :N2]) + gamma * np.eye(N2),
        np.multiply(KX[:N2, :N2], KT_pi[:N2, :N2]),
    )
    mu_pi_prime_split1 = np.linalg.solve(
        np.multiply(KX[:N2, :N2], KT[:N2, :N2]) + gamma * np.eye(N2),
        np.multiply(KX[:N2, :N2], KT_pi_prime[:N2, :N2]),
    )

    mu_logging_split2 = np.linalg.solve(
        np.multiply(KX[N2:, N2:], KT[N2:, N2:]) + gamma * np.eye(N2),
        np.multiply(KX[N2:, N2:], KT[N2:, N2:]),
    )
    mu_pi_split2 = np.linalg.solve(
        np.multiply(KX[N2:, N2:], KT[N2:, N2:]) + gamma * np.eye(N2),
        np.multiply(KX[N2:, N2:], KT_pi[N2:, N2:]),
    )
    mu_pi_prime_split2 = np.linalg.solve(
        np.multiply(KX[N2:, N2:], KT[N2:, N2:]) + gamma * np.eye(N2),
        np.multiply(KX[N2:, N2:], KT_pi_prime[N2:, N2:]),
    )

    # DR
    left_side = (
        mu_pi_prime_split1
        - mu_pi_split1
        + (w_pi_prime_split1 - w_pi_split1) * (np.eye(N2) - mu_logging_split1)
    )
    right_side = (
        mu_pi_prime_split2
        - mu_pi_split2
        + (w_pi_prime_split2 - w_pi_split2) * (np.eye(N2) - mu_logging_split2)
    )

    KY = pairwise_kernels(Y_split1, Y_split2, metric=kernel_function, **kwargs)
    prod = left_side.T @ KY @ right_side

    U = prod.mean(1)
    return np.sqrt(len(U)) * U.mean() / U.std()