# %%
# Imports
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import pandas as pd
import time
from sklearn.metrics import pairwise_distances
from kpt import kernel_two_sample_test_reweight
from dr_kpt import xMMD2dr, xMMD2dr_cross_fit
from warfarin import make_scenario, generate_ope_data, find_best_params
import os
from tqdm import tqdm

scenario_id = "I"
seed = 42
X, T, Y, pi, pi_prime = make_scenario(scenario_id, seed=seed)
# X = X[:ns]
data = generate_ope_data(X, T, Y, pi, pi_prime)

Y = data["Y"].reshape(-1, 1)
T = data["T"]
w_pi = data["w_pi"]
w_pi_prime = data["w_pi_prime"]
pi_samples = data["pi_samples"]
pi_prime_samples = data["pi_prime_samples"]

reg_lambda = find_best_params(X, T, Y)
try:
    sigma2 = np.median(pairwise_distances(Y, Y)) ** 2
    gamma_k = 1.0 / sigma2
except:
    gamma_k = None

stat = xMMD2dr_cross_fit(
    Y,
    X,
    T,
    w_pi,
    w_pi_prime,
    pi_samples,
    pi_prime_samples,
    kernel_function="rbf",
    reg_lambda=0.1,
)
pval = 1 - st.norm.cdf(stat)

# %%
