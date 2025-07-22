# %%
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from scipy.stats import norm, bernoulli
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

from dr_kpt import xMMD2dr_cross_fit
from kpt import kernel_two_sample_test_reweight  # your reweighted baseline
from environment import find_best_params

# %%
# === Setup ===
name_folder = "data_ihdp/"
os.makedirs(name_folder, exist_ok=True)
size_subset = 500
num_experiments = 20
iterations = 1000

b_list = ["I", "II", "III", "IV"]
method_list = ["PE-linear", "KPE", "DR-CF"]
np.random.seed(0)

# === Load IHDP ===
df = pd.read_csv("ihdp.csv", index_col=0)
covs_cont = [
    "bw",
    "momage",
    "nnhealth",
    "birth.o",
    "parity",
    "moreprem",
    "cigs",
    "alcohol",
    "ppvt.imp",
]
covs_cat = [
    "bwg",
    "female",
    "mlt.birt",
    "b.marry",
    "livwho",
    "language",
    "whenpren",
    "drugs",
    "othstudy",
]
features = covs_cont + covs_cat

df1 = df[features + ["iqsb.36", "treat"]].dropna()
scaler = StandardScaler()
df1[covs_cont] = scaler.fit_transform(df1[covs_cont])

X_all = df1[features].to_numpy()
T_all = df1["treat"].to_numpy()
Y_all = df1["iqsb.36"].to_numpy() / np.linalg.norm(df1["iqsb.36"].to_numpy())


# %%


def make_policy(X, beta=None, offset=0.0):
    X = np.atleast_2d(X)
    if beta is None or np.isscalar(beta):
        beta = np.linspace(0.1, 0.5, X.shape[1])
    logits = X @ beta + offset
    prob_1 = 1 / (1 + np.exp(-logits))
    return np.stack([1 - prob_1, prob_1], axis=1)


def make_policy_mixture(X, beta, delta=2.0, lambda_=0.5):
    """
    Returns a policy with the same mean as the base policy,
    but greater variance, via a symmetric mixture of logits.
    """
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
        return lambda_ * np.stack([1 - p1_corrected, p1_corrected], axis=1) + (
            1 - lambda_
        ) * np.stack([1 - p2_corrected, p2_corrected], axis=1)

    return policy


def make_scenario_binary(scenario_id, X, beta_base=None):
    """
    Returns pi_fn, pi_prime_fn: functions mapping X to [p(0), p(1)] for binary actions.
    """
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
        # Same policy — identical distributions
        pi_fn = bernoulli_policy(beta_base, offset=1.0)
        pi_prime_fn = bernoulli_policy(beta_base, offset=1.0)

    elif scenario_id == "II":
        # Mean shift only
        pi_fn = bernoulli_policy(beta_base, offset=0.0)
        pi_prime_fn = lambda X: 1 - bernoulli_policy(beta_base, offset=0.0)(X)


    elif scenario_id == "III":
        # Same mean, higher variance
        pi_fn = bernoulli_policy(beta_base, offset=0.0)
        pi_prime_fn = make_policy_mixture(X, beta_base, delta=2.0, lambda_=0.2)


    elif scenario_id == "IV":
        # Both mean and higher moments differ (asymmetric mixture)
        pi_fn = bernoulli_policy(beta_base, offset=0.5)
        pi_prime_fn = lambda X: 0.5 * bernoulli_policy(beta_base, 1.0)(X) + 0.5 * bernoulli_policy(beta_base, -1.0)(X)


    else:
        raise ValueError(f"Unknown scenario {scenario_id}")

    return pi_fn, pi_prime_fn

# %%
rng = np.random.RandomState(42)

# %%
size_subset = 800
idx = rng.choice(len(X_all), size_subset, replace=False)
X = X_all[idx]
T = T_all[idx]
Y = Y_all[idx][:, None]

# %%
# Logging policy
logreg = LogisticRegression(C=1e4, max_iter=1000, random_state=42).fit(X, T)
pi0_scores = logreg.predict_proba(X)[:, 1]
pi0_scores = np.clip(pi0_scores, 1e-4, 1 - 1e-4)
pi0_probs = np.stack([1 - pi0_scores, pi0_scores], axis=1)

# %%
logreg.coef_[0]


# %%
b = "III"
# Target policies
pi_fn, pi_prime_fn = make_scenario_binary(b, X, logreg.coef_[0])

pi_probs = pi_fn(X)
pi_prime_probs = pi_prime_fn(X)

# %%

w_pi = pi_probs[np.arange(len(T)), T] / pi0_probs[np.arange(len(T)), T]
w_pi_prime = pi_prime_probs[np.arange(len(T)), T] / pi0_probs[np.arange(len(T)), T]

# %%
pi_samples = np.array([rng.choice([0, 1], p=p) for p in pi_probs])
pi_prime_samples = np.array([rng.choice([0, 1], p=p) for p in pi_prime_probs])

sigma2 = np.median(pairwise_distances(Y, Y)) ** 2
gamma_k = 1.0 / sigma2
# %%
pi_samples.mean(), pi_prime_samples.mean()
# %%
np.var(pi_samples), np.var(pi_prime_samples)
# %%
stat, _, pval = kernel_two_sample_test_reweight(
    Y,
    w_pi,
    w_pi_prime,
    kernel_function="linear",
    iterations=iterations,
    random_state=42,
)
pval
#%%
stat, _, pval = kernel_two_sample_test_reweight(
    Y,
    w_pi,
    w_pi_prime,
    kernel_function="rbf",
    iterations=iterations,
    gamma=gamma_k,
    random_state=42,
)
pval
# %%
stat = xMMD2dr_cross_fit(
    Y=Y,
    X=X,
    logging_T=T,
    w_pi=w_pi,
    w_pi_prime=w_pi_prime,
    pi_samples=pi_samples,
    pi_prime_samples=pi_prime_samples,
    kernel_function="rbf",
    reg_lambda=1e2,
    gamma=gamma_k,
)
pval = 1 - norm.cdf(stat)
pval
# %%

# method_list = ["DR-CF"]
method_list = ["PE-linear", "KPE", "DR-CF"]


# === Main Loop ===
for b in b_list:
    for method in method_list:
        p_values = np.zeros(num_experiments)
        stats = np.zeros(num_experiments)
        times = np.zeros(num_experiments)

        for n in tqdm(range(num_experiments), desc=f"Scenario {b}, Method {method}"):
            idx = rng.choice(len(X_all), size_subset, replace=False)
            X = X_all[idx]
            T = T_all[idx]
            Y = Y_all[idx][:, None]

            # Logging policy
            logreg = LogisticRegression(max_iter=1000, random_state=42).fit(X, T)
            pi0_scores = logreg.predict_proba(X)[:, 1]
            pi0_scores = np.clip(pi0_scores, 1e-4, 1 - 1e-4)
            pi0_probs = np.stack([1 - pi0_scores, pi0_scores], axis=1)

            # Target policies
            pi_fn, pi_prime_fn = make_scenario_binary(b, X, logreg.coef_[0])

            pi_probs = pi_fn(X)
            pi_prime_probs = pi_prime_fn(X)

            w_pi = pi_probs[np.arange(len(T)), T] / pi0_probs[np.arange(len(T)), T]
            w_pi_prime = (
                pi_prime_probs[np.arange(len(T)), T] / pi0_probs[np.arange(len(T)), T]
            )

            pi_samples = np.array([rng.choice([0, 1], p=p) for p in pi_probs])
            pi_prime_samples = np.array(
                [rng.choice([0, 1], p=p) for p in pi_prime_probs]
            )

            try:
                sigma2 = np.median(pairwise_distances(Y, Y)) ** 2
                gamma_k = 1.0 / sigma2
            except:
                gamma_k = None

            try:
                t0 = time.time()
                if method == "PE-linear":
                    stat, _, pval = kernel_two_sample_test_reweight(
                        Y,
                        w_pi,
                        w_pi_prime,
                        kernel_function="linear",
                        iterations=iterations,
                        random_state=n,
                    )
                elif method == "KPE":
                    stat, _, pval = kernel_two_sample_test_reweight(
                        Y,
                        w_pi,
                        w_pi_prime,
                        kernel_function="rbf",
                        gamma=gamma_k,
                        iterations=iterations,
                        random_state=n,
                    )
                elif method == "DR-CF":
                    # reg_lambda = find_best_params(X, T, Y)
                    stat = xMMD2dr_cross_fit(
                        Y=Y,
                        X=X,
                        logging_T=T,
                        w_pi=w_pi,
                        w_pi_prime=w_pi_prime,
                        pi_samples=pi_samples,
                        pi_prime_samples=pi_prime_samples,
                        kernel_function="rbf",
                        reg_lambda=1e2,
                        gamma=gamma_k,
                    )
                    pval = 1 - norm.cdf(stat)
                    print(method, pval)
                else:
                    raise ValueError("Unknown method")

                elapsed = time.time() - t0
            except Exception:
                stat, pval, elapsed = np.nan, np.nan, 0.0

            p_values[n] = pval
            stats[n] = stat
            times[n] = elapsed

        df = pd.DataFrame({"p_value": p_values, "stat": stats, "time": times})
        fname = f"{name_folder}/b{b}_{method}.csv"
        df.to_csv(fname, index=False)
        print(f"Saved: {fname}")

# %%
