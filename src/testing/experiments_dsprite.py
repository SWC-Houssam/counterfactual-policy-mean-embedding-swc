# dsprite_experiment.py
import argparse
import numpy as np
import pandas as pd
import os
import time
import scipy.stats as st
from sklearn.metrics import pairwise_distances
from kpt import kernel_two_sample_test_reweight
from dr_kpt import xMMD2dr_cross_fit
from dsprite import run_dsprite_experiment  # assumes your main logic is there

EXPNAME = "dsprite_experiment"
RESULT_DIR = f"results/{EXPNAME}/"
PARAM_CSV = f"experiment_parameters/{EXPNAME}_parameters.csv"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs("experiment_parameters", exist_ok=True)
NB_SEEDS = 100


def run_single_experiment(scenario_id, method, seed, iterations=1000):
    data = run_dsprite_experiment(scenario_id=scenario_id, seed=seed)

    Y = data["Y"].reshape(data["Y"].shape[0], -1)  # flatten images
    X = data["U"]
    T = data["A"]
    w_pi = data["w_pi"]
    w_pi_prime = data["w_pi_prime"]
    pi_samples = data["pi_samples"]
    pi_prime_samples = data["pi_prime_samples"]

    try:
        sigma2 = np.median(pairwise_distances(Y, Y)) ** 2
        gamma_k = 1.0 / sigma2
    except:
        gamma_k = None

    try:
        t0 = time.time()
        if method == "KPT-linear":
            stat, _, pval = kernel_two_sample_test_reweight(
                Y,
                w_pi,
                w_pi_prime,
                kernel_function="linear",
                iterations=iterations,
                random_state=seed,
            )
        elif method == "KPT-rbf":
            stat, _, pval = kernel_two_sample_test_reweight(
                Y,
                w_pi,
                w_pi_prime,
                kernel_function="rbf",
                iterations=iterations,
                random_state=seed,
            )
        elif method == "KPT-poly":
            stat, _, pval = kernel_two_sample_test_reweight(
                Y,
                w_pi,
                w_pi_prime,
                kernel_function="polynomial",
                iterations=iterations,
                random_state=seed,
            )
        elif method == "DR-KPT-rbf":
            stat = xMMD2dr_cross_fit(
                Y,
                X,
                T,
                w_pi,
                w_pi_prime,
                pi_samples,
                pi_prime_samples,
                kernel_function="rbf",
                reg_lambda=1e2,
                gamma=gamma_k,
            )
            pval = 1 - st.norm.cdf(stat)
        elif method == "DR-KPT-poly":
            stat = xMMD2dr_cross_fit(
                Y,
                X,
                T,
                w_pi,
                w_pi_prime,
                pi_samples,
                pi_prime_samples,
                kernel_function="polynomial",
                reg_lambda=1e2,
            )
            pval = 1 - st.norm.cdf(stat)
        else:
            raise ValueError(f"Unknown method: {method}")
        elapsed = time.time() - t0
    except Exception as e:
        stat, pval, elapsed = np.nan, np.nan, 0.0

    result = {
        "p_value": pval,
        "stat": stat,
        "time": elapsed,
        "scenario": scenario_id,
        "method": method,
        "seed": seed,
    }
    return result


def get_parameters_experiment():
    scenario_list = ["I", "III", "IV"]
    method_list = ["KPT-linear", "KPT-rbf", "KPT-poly", "DR-KPT-rbf", "DR-KPT-poly"]
    seeds = list(range(NB_SEEDS))

    with open(PARAM_CSV, "w") as f:
        f.write("line,scenario,method,seed\n")
        line = 0
        for scenario in scenario_list:
            for method in method_list:
                for seed in seeds:
                    f.write(f"{line},{scenario},{method},{seed}\n")
                    line += 1
    print(f"Parameter file written to {PARAM_CSV}")


def run_from_arguments(args):
    result = run_single_experiment(
        scenario_id=args.scenario, method=args.method, seed=args.seed
    )
    df = pd.DataFrame([result])

    # One file per scenario and method, all seeds appended
    fname = f"{RESULT_DIR}/scenario{args.scenario}_{args.method}.csv"
    if os.path.exists(fname):
        df.to_csv(fname, mode="a", index=False, header=False)
    else:
        df.to_csv(fname, index=False)
    print(f"Appended result to: {fname}")



def get_results_table():
    name_folder = RESULT_DIR
    alpha = 0.05
    scenario_list = ["I", "III", "IV"]
    method_list = ["KPT-linear", "KPT-rbf", "KPT-poly", "DR-KPT-rbf", "DR-KPT-poly"]

    rejection_matrix = np.zeros((len(scenario_list), len(method_list)))

    for i, scenario in enumerate(scenario_list):
        for j, method in enumerate(method_list):
            fname = f"{name_folder}/scenario{scenario}_{method}.csv"
            try:
                df = pd.read_csv(fname)
                p_values = df["p_value"].dropna()
                if len(p_values) > 0:
                    rejection_matrix[i, j] = (p_values < alpha).mean()
            except FileNotFoundError:
                print(f"File not found: {fname}")
                continue

    df_reject = pd.DataFrame(rejection_matrix, index=scenario_list, columns=method_list)
    print(df_reject)

    latex_path = os.path.join(name_folder, "rejection_table.tex")
    with open(latex_path, "w") as f:
        f.write(df_reject.T.to_latex(float_format="{:0.2f}".format))
    print(f"LaTeX table saved to {latex_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run single experiment")
    parser.add_argument(
        "--get_parameters_experiment",
        action="store_true",
        help="Generate CSV parameter file",
    )
    parser.add_argument("--results", action="store_true", help="Aggregate all results")
    parser.add_argument("--scenario", type=str, help="Scenario ID (I, III, IV)")
    parser.add_argument(
        "--method",
        type=str,
        choices=["KPT-linear", "KPT-rbf", "KPT-poly", "DR-KPT-rbf", "DR-KPT-poly"],
        help="Estimator method",
    )
    parser.add_argument("--seed", type=int, help="Seed for the experiment")

    args = parser.parse_args()

    if args.get_parameters_experiment:
        get_parameters_experiment()

    if args.run:
        run_from_arguments(args)

    if args.results:
        get_results_table()
