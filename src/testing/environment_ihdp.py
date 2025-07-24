# ihdp_environment.py

import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# === Setup ===
name_folder = "data_ihdp/"
os.makedirs(name_folder, exist_ok=True)

# === Load IHDP ===
df = pd.read_csv("ihdp.csv", index_col=0)
covs_cont = [
    "bw", "momage", "nnhealth", "birth.o", "parity", "moreprem",
    "cigs", "alcohol", "ppvt.imp"
]
covs_cat = [
    "bwg", "female", "mlt.birt", "b.marry", "livwho",
    "language", "whenpren", "drugs", "othstudy"
]
features = covs_cont + covs_cat

df1 = df[features + ["iqsb.36", "treat"]].dropna()
scaler = StandardScaler()
df1[covs_cont] = scaler.fit_transform(df1[covs_cont])

X_all = df1[features].to_numpy()
T_all = df1["treat"].to_numpy()
Y_all = df1["iqsb.36"].to_numpy()
Y_all /= np.linalg.norm(Y_all)

def make_policy(X, beta=None, offset=0.0):
    X = np.atleast_2d(X)
    if beta is None or np.isscalar(beta):
        beta = np.linspace(0.1, 0.5, X.shape[1])
    logits = X @ beta + offset
    prob_1 = 1 / (1 + np.exp(-logits))
    return np.stack([1 - prob_1, prob_1], axis=1)

def make_policy_mixture(X, beta, delta=2.0, lambda_=0.5):
    p_base = 1 / (1 + np.exp(-X @ beta))
    logits1 = X @ beta + delta
    logits2 = X @ beta - delta
    p1 = 1 / (1 + np.exp(-logits1))
    p2 = 1 / (1 + np.exp(-logits2))
    mean_mixture = lambda_ * p1 + (1 - lambda_) * p2
    correction = p_base / mean_mixture
    p1_corrected = np.clip(p1 * correction, 1e-3, 1 - 1e-3)
    p2_corrected = np.clip(p2 * correction, 1e-3, 1 - 1e-3)

    def policy(X_):
        return lambda_ * np.stack([1 - p1_corrected, p1_corrected], axis=1) + \
               (1 - lambda_) * np.stack([1 - p2_corrected, p2_corrected], axis=1)

    return policy

def make_scenario_binary(scenario_id, X, beta_base=None):
    d = X.shape[1]
    if beta_base is None:
        beta_base = np.ones(d) / np.sqrt(d)

    def bernoulli_policy(w, offset=0.0):
        def get_probs(X):
            logits = X @ w + offset
            probs_1 = 1 / (1 + np.exp(-logits))
            return np.stack([1 - probs_1, probs_1], axis=1)
        return get_probs

    if scenario_id == "I":
        pi_fn = bernoulli_policy(beta_base, offset=1.0)
        pi_prime_fn = bernoulli_policy(beta_base, offset=1.0)

    elif scenario_id == "II":
        pi_fn = bernoulli_policy(beta_base, offset=0.0)
        pi_prime_fn = lambda X: 1 - bernoulli_policy(beta_base, offset=0.0)(X)

    # elif scenario_id == "III":
    #     pi_fn = bernoulli_policy(beta_base, offset=0.0)
    #     pi_prime_fn = make_policy_mixture(X, beta_base, delta=2.0, lambda_=0.2)

    elif scenario_id == "IV":
        # pi_fn = bernoulli_policy(beta_base)
        # pi_prime_fn = lambda X: 0.5 * bernoulli_policy(beta_base, 1.0)(X) + \
        #                         0.5 * bernoulli_policy(beta_base, -1.0)(X)
        pi_fn = bernoulli_policy(beta_base, offset=0.0)
        pi_prime_fn = lambda X: 1 - bernoulli_policy(beta_base, offset=1.0)(X)

    else:
        raise ValueError(f"Unknown scenario {scenario_id}")

    return pi_fn, pi_prime_fn
