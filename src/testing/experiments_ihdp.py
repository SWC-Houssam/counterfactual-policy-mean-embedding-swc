# experiments_ihdp.py

import argparse
import numpy as np
import pandas as pd
import os
import time
import scipy.stats as st
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LogisticRegression

from kpt import kernel_two_sample_test_reweight
from dr_kpt import xMMD2dr_cross_fit
from environment_ihdp import X_all, T_all, Y_all, make_scenario_binary

EXPNAME = "ihdp_experiment"
RESULT_DIR = f"results/{EXPNAME}/"
PARAM_CSV = f"experiment_parameters/{EXPNAME}_parameters.csv"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs("experiment_parameters", exist_ok=True)
NB_SEEDS = 100
SAMPLE_SIZE = 500
ITERATIONS = 10000

METHODS = ["KPT-linear", "KPT-rbf", "DR-KPT"]
SCENARIOS = ["I", "II", "IV"]


def run_single_experiment(scenario_id, method, seed):
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X_all), SAMPLE_SIZE, replace=False)
    X = X_all[idx]
    T = T_all[idx]
    Y = Y_all[idx][:, None]

    logreg = LogisticRegression(max_iter=1000, random_state=seed).fit(X, T)
    pi0_scores = logreg.predict_proba(X)[:, 1]
    pi0_scores = np.clip(pi0_scores, 1e-4, 1 - 1e-4)
    pi0_probs = np.stack([1 - pi0_scores, pi0_scores], axis=1)

    pi_fn, pi_prime_fn = make_scenario_binary(scenario_id, X, logreg.coef_[0])
    pi_probs = pi_fn(X)
    pi_prime_probs = pi_prime_fn(X)

    w_pi = pi_probs[np.arange(len(T)), T] / pi0_probs[np.arange(len(T)), T]
    w_pi_prime = pi_prime_probs[np.arange(len(T)), T] / pi0_probs[np.arange(len(T)), T]

    pi_samples = np.array([rng.choice([0, 1], p=p) for p in pi_probs])
    pi_prime_samples = np.array([rng.choice([0, 1], p=p) for p in pi_prime_probs])

    try:
        sigma2 = np.median(pairwise_distances(Y, Y)) ** 2
        gamma_k = 1.0 / sigma2
    except:
        gamma_k = None

    try:
        t0 = time.time()
        if method == "KPT-linear":
            stat, _, pval = kernel_two_sample_test_reweight(
                Y, w_pi, w_pi_prime, kernel_function="linear",
                iterations=ITERATIONS, random_state=seed)

        elif method == "KPT-rbf":
            stat, _, pval = kernel_two_sample_test_reweight(
                Y, w_pi, w_pi_prime, kernel_function="rbf",
                gamma=gamma_k, iterations=ITERATIONS, random_state=seed)

        elif method == "DR-KPT":
            stat = xMMD2dr_cross_fit(
                Y, X, T, w_pi, w_pi_prime,
                pi_samples, pi_prime_samples,
                kernel_function="rbf", reg_lambda=1e-2, gamma=gamma_k)
            pval = 1 - st.norm.cdf(stat)

        else:
            raise ValueError("Unknown method")

        elapsed = time.time() - t0
    except Exception:
        stat, pval, elapsed = np.nan, np.nan, 0.0

    return {
        "p_value": pval,
        "stat": stat,
        "time": elapsed,
        "scenario": scenario_id,
        "method": method,
        "seed": seed,
    }


def get_parameters_experiment():
    with open(PARAM_CSV, "w") as f:
        f.write("line,scenario,method,seed\n")
        line = 0
        for scenario in SCENARIOS:
            for method in METHODS:
                for seed in range(NB_SEEDS):
                    f.write(f"{line},{scenario},{method},{seed}\n")
                    line += 1
    print(f"Parameter file written to {PARAM_CSV}")


def run_from_arguments(args):
    result = run_single_experiment(
        scenario_id=args.scenario,
        method=args.method,
        seed=args.seed
    )
    df = pd.DataFrame([result])
    fname = f"{RESULT_DIR}/scenario{args.scenario}_{args.method}_seed{args.seed}.csv"
    df.to_csv(fname, index=False)
    print(f"Saved: {fname}")


def get_results_table():
    alpha = 0.05
    rejection_matrix = np.zeros((len(SCENARIOS), len(METHODS)))

    for i, scenario in enumerate(SCENARIOS):
        for j, method in enumerate(METHODS):
            rejections = []
            for seed in range(NB_SEEDS):
                fname = f"{RESULT_DIR}/scenario{scenario}_{method}_seed{seed}.csv"
                try:
                    df = pd.read_csv(fname)
                    rejections.append((df["p_value"] < alpha).mean())
                except FileNotFoundError:
                    continue
            if rejections:
                rejection_matrix[i, j] = np.mean(rejections)

    df_reject = pd.DataFrame(rejection_matrix, index=SCENARIOS, columns=METHODS)
    print(df_reject)

    latex_path = os.path.join(RESULT_DIR, "rejection_table.tex")
    with open(latex_path, "w") as f:
        f.write(df_reject.T.to_latex(float_format="{:0.2f}".format))
    print(f"LaTeX table saved to {latex_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--get_parameters_experiment", action="store_true")
    parser.add_argument("--results", action="store_true")
    parser.add_argument("--scenario", type=str, help="Scenario ID (I, II, III, IV)")
    parser.add_argument("--method", type=str, choices=METHODS, help="Estimator method")
    parser.add_argument("--seed", type=int, help="Seed")

    args = parser.parse_args()

    if args.get_parameters_experiment:
        get_parameters_experiment()

    if args.run:
        run_from_arguments(args)

    if args.results:
        get_results_table()
