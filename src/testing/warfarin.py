import matplotlib.pyplot as plt

# from off_pol_eval_functions import *
import sys
import numpy as np
import scipy
import scipy.stats
from scipy.spatial.distance import cdist
import csv
from sklearn import tree
from sklearn import linear_model, neighbors, ensemble, tree
from scipy.stats import laplace, bernoulli
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV

# from warfarin_functions import *

ntrs = np.arange(100, 2501, 100)
runsper = 4
runsperper = 25
params = [ntr for ntr in ntrs for run in range(runsper)]

myid = 0  # int(sys.argv[1])
# np.random.seed(myid)
ntr = 1000  # params[myid-1]

f = open("warfrin.csv", "rU")
csvr = csv.reader(f, dialect=csv.excel)
header = np.array(next(csvr))
data = list(csvr)
f.close()

# filter only to subjects that reached stable dose of warfarin
# and stable observed INR
data = [x for x in data if x[37] == "1" and x[38] != "NA" and x[39] != "NA"]

agegroups = sorted(set(x[8].replace("NA", "") for x in data))

# %%
xmap = (
    lambda x: [
        # gender 0
        ("Male?", x[3] == "male"),
        # race 1:4
        ("White?", x[5] == "White"),
        ("Asian?", x[5] == "Asian"),
        ("Black?", x[5] == "Black or African American"),
        # ethnicity 4
        ("Non-hispanic?", x[7] == "not Hispanic or Latino"),
        # age 5:7
        ("Age group", agegroups.index(x[8]) if x[8] in agegroups else 0),
        ("No age?", x[8] not in agegroups or agegroups.index(x[8]) == 0),
        # height 7
        ("Height", float(x[9]) if x[9] not in ("NA", "") else 0.0),
        # x[9] in ('NA', ''), # NA indicator
        # weight 8
        ("Weight", float(x[10]) if x[10] not in ("NA", "") else 0.0),
        # x[10] in ('NA', ''), # NA indicator
        # BMI
        (
            "BMI",
            (
                float(x[10]) * 100.0 * 100.0 / float(x[9]) / float(x[9])
                if x[10] not in ("NA", "") and x[9] not in ("NA", "")
                else 0.0
            ),
        ),
    ]
    + [  # Indication for Warfarin Treatment 9:17
        ("Indication for Warfarin Treatment: " + str(i), str(i) in x[11])
        for i in range(1, 9)
        # ] + [# comorbidities
        #     c in (y.strip() for y in x[12].lower().split('; '))
        #     for c in comorbidities
    ]
    + [  # diabetes 17:19
        ("Diabetes=0?", x[13] == "0"),
        ("Diabetes=1?", x[13] == "1"),
    ]
    + [  # Congestive Heart Failure and/or Cardiomyopathy 19:21
        ("Congestive Heart Failure and/or Cardiomyopathy=0?", x[14] == "0"),
        ("Congestive Heart Failure and/or Cardiomyopathy=1?", x[14] == "1"),
    ]
    + [  # Valve Replacement 21:23
        ("Valve Replacement=0?", x[15] == "0"),
        ("Valve Replacement=1?", x[15] == "1"),
        # ] + [# medications
        #   x[16]
    ]
    + [
        # Aspirin 24:26
        ("aspirin=0?", x[17] == "0"),
        ("aspirin=1?", x[17] == "1"),
        # Acetaminophen or Paracetamol (Tylenol) 26:28
        ("Acetaminophen=0?", x[18] == "0"),
        ("Acetaminophen=1?", x[18] == "1"),
        # Was Dose of Acetaminophen or Paracetamol (Tylenol) >1300mg/day 28:30
        ("Acetaminophen hi dose=0?", x[19] == "0"),
        ("Acetaminophen hi dose=1?", x[19] == "1"),
        # Simvastatin (Zocor) 30:32
        ("Simvastatin=0?", x[20] == "0"),
        ("Simvastatin=1?", x[20] == "1"),
        # Atorvastatin (Lipitor) 32:34
        ("Simvastatin=0?", x[21] == "0"),
        ("Simvastatin=1?", x[21] == "1"),
        # Fluvastatin (Lescol) 34:36
        ("Fluvastatin=0?", x[22] == "0"),
        ("Fluvastatin=1?", x[22] == "1"),
        # Lovastatin (Mevacor) 36:38
        ("Lovastatin=0?", x[23] == "0"),
        ("Lovastatin=1?", x[23] == "1"),
        # Pravastatin (Pravachol) 38:40
        ("Pravastatin=0?", x[24] == "0"),
        ("Pravastatin=1?", x[24] == "1"),
        # Rosuvastatin (Crestor) 40:42
        ("Rosuvastatin=0?", x[25] == "0"),
        ("Rosuvastatin=1?", x[25] == "1"),
        # Cerivastatin (Baycol) 42:43
        ("Cerivastatin=0?", x[26] == "0"),
        ("Cerivastatin=1?", x[26] == "1"),
        # Amiodarone (Cordarone)
        ("Amiodarone=0?", x[27] == "0"),
        ("Amiodarone=1?", x[27] == "1"),
        # Carbamazepine (Tegretol)
        ("Carbamazepine=0?", x[28] == "0"),
        ("Carbamazepine=1?", x[28] == "1"),
        # Phenytoin (Dilantin)
        ("Phenytoin=0?", x[29] == "0"),
        ("Phenytoin=1?", x[29] == "1"),
        # Rifampin or Rifampicin
        ("Rifampin=0?", x[30] == "0"),
        ("Rifampin=1?", x[30] == "1"),
        # Sulfonamide Antibiotics
        ("Sulfonamide Antibiotics=0?", x[31] == "0"),
        ("Sulfonamide Antibiotics=1?", x[31] == "1"),
        # Macrolide Antibiotics
        ("Macrolide Antibiotics=0?", x[32] == "0"),
        ("Macrolide Antibiotics=1?", x[32] == "1"),
        # Anti-fungal Azoles
        ("Anti-fungal Azoles=0?", x[33] == "0"),
        ("Anti-fungal Azoles=1?", x[33] == "1"),
        # Herbal Medications, Vitamins, Supplements
        ("Herbal Medications, Vitamins, Supplements=0?", x[34] == "0"),
        ("Herbal Medications, Vitamins, Supplements=1?", x[34] == "1"),
    ]
    + [
        # smoker
        ("Smoker=0?", x[40] == "0"),
        ("Smoker=0?", x[40] == "1"),
    ]
    + [
        # CYP2C9 consensus
        ("CYP2C9 *1/*1", x[59] == "*1/*1"),
        ("CYP2C9 *1/*2", x[59] == "*1/*2"),
        ("CYP2C9 *1/*3", x[59] == "*1/*3"),
        ("CYP2C9 NA", x[59] == "" or x[59] == "NA"),
        # VKORC1 -1639 consensus
        ("VKORC1 -1639 A/A", x[60] == "A/A"),
        ("VKORC1 -1639 A/G", x[60] == "A/G"),
        ("VKORC1 -1639 G/G", x[60] == "G/G"),
        # VKORC1 497 consensus
        ("VKORC1 497 T/T", x[61] == "T/T"),
        ("VKORC1 497 G/T", x[61] == "G/T"),
        ("VKORC1 497 G/G", x[61] == "G/G"),
        # VKORC1 1173 consensus
        ("VKORC1 1173 T/T", x[62] == "T/T"),
        ("VKORC1 1173 C/T", x[62] == "C/T"),
        ("VKORC1 1173 C/C", x[62] == "C/C"),
        # VKORC1 1542 consensus
        ("VKORC1 1542 C/C", x[63] == "C/C"),
        ("VKORC1 1542 C/G", x[63] == "C/G"),
        ("VKORC1 1542 G/G", x[63] == "G/G"),
        # VKORC1 3730 consensus
        ("VKORC1 3730 A/A", x[64] == "A/A"),
        ("VKORC1 3730 A/G", x[64] == "A/G"),
        ("VKORC1 3730 G/G", x[64] == "G/G"),
        # VKORC1 2255 consensus
        ("VKORC1 2255 C/C", x[65] == "C/C"),
        ("VKORC1 2255 C/T", x[65] == "C/T"),
        ("VKORC1 2255 T/T", x[65] == "T/T"),
        # VKORC1 -4451 consensus
        ("VKORC1 -4451 C/C", x[66] == "C/C"),
        ("VKORC1 -4451 A/C", x[66] == "A/C"),
        ("VKORC1 -4451 A/A", x[66] == "A/A"),
        #     ,
        #     ('Therapeutic Dose',float(x[38]) if x[38] not in ('NA', '') else 0.),
        #     ('INR On Therapeutic Dose',float(x[39]) if x[39] not in ('NA', '') else 0.),
        #     ('Target INR', float(x[35]) if x[35] not in ('NA', '') else 0. )
    ]
)
# %%
X = np.array([list(zip(*xmap(x)))[1] for x in data])
Xnames = np.array(list(zip(*xmap(data[0])))[0])

goodidx = np.where(X.std(axis=0) >= 0.05)[0]
X = X[:, goodidx]
Xnames = Xnames[goodidx]

# Filter out by where BMI is nonzero (assumes BMI is at column index 9 after filtering)
goodbmi = np.where(X[:, 9] > 0.003)[0]
X = X[goodbmi, :]
n = X.shape[0]

# Extract target and observed INR values, handling missing values
target_INR = np.array([float(x[35]) if x[35] not in ("NA", "") else 0.0 for x in data])
therapeut_dose = np.array(
    [float(x[38]) if x[38] not in ("NA", "") else 0.0 for x in data]
)
obs_INR = np.array([float(x[39]) if x[39] not in ("NA", "") else 0.0 for x in data])

# Apply same goodbmi filtering
target_INR = target_INR[goodbmi]
therapeut_dose = therapeut_dose[goodbmi]
obs_INR = obs_INR[goodbmi]

# Global values
mu_dose = np.mean(therapeut_dose)
std_dose = np.std(therapeut_dose)
mu_bmi = np.mean(X[:, 9])
std_bmi = np.std(X[:, 9])

# regr = linear_model.LinearRegression()
# regr.fit(X[:,9].reshape([n,1]), therapeut_dose.reshape([n,1]))

theta = 0.5
eps = np.random.normal(size=n)

# eps = np.random.randn(n)
bmi_Z = (X[:, 9] - mu_bmi) / std_bmi

# fit on centered model
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(bmi_Z.reshape([n, 1]), therapeut_dose.reshape([n, 1]))


T = mu_dose + (np.sqrt(theta) * bmi_Z * std_dose + np.sqrt(1 - theta) * std_dose * eps)
# sim_dose[sim_dose < 0 ] = - sim_dose[sim_dose < 0 ]


# %%
def simulated_loss(sim_dose, therapeut_dose):
    loss = np.maximum(
        np.abs(therapeut_dose - sim_dose) - 0.1 * therapeut_dose,
        np.zeros_like(therapeut_dose),
    )
    return loss + np.random.normal(scale=0.1, size=therapeut_dose.shape)


Y = np.asarray(simulated_loss(T, therapeut_dose))


# %%
class GaussianPolicy:
    def __init__(self, w, scale=1.0):
        self.w = w
        self.scale = scale

    def sample_treatments(self, X):
        return np.random.normal(self.get_mean(X), self.scale)

    def get_mean(self, X):
        return X @ self.w

    def get_propensities(self, X, t):
        mean = self.get_mean(X)
        return (1 / (self.scale * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((t - mean) / self.scale) ** 2
        )


class EstimatedLoggingPolicy:
    def __init__(self, X, T, variance=1.0):
        self.model_type = "continuous"
        self.model = LinearRegression()
        self.model.fit(X, T)
        self.variance = variance

    def get_propensities(self, X, t):
        mean = self.model.predict(X)
        var = self.variance
        coef = 1 / np.sqrt(2 * np.pi * var)
        exp_term = np.exp(-0.5 * ((t - mean) ** 2) / var)
        return coef * exp_term

    def get_mean(self, X):
        return self.model.predict(X)


def generate_ope_data(X, T, Y, policy_pi, policy_pi_prime):
    """
    Simulate logged bandit data and prepare inputs for counterfactual evaluation.

    Parameters:
        X                : Covariates (n, d)
        policy_logging   : Policy that generated the logged data (π₀)
        policy_pi        : Target policy π
        policy_pi_prime  : Alternative policy π′
        beta, gamma      : Outcome model parameters
        noise_std        : Std of additive Gaussian noise

    Returns:
        dict with {X, T, Y, w_pi, w_pi_prime, pi_samples, pi_prime_samples}
    """

    estimate_logging_propensities = EstimatedLoggingPolicy(
        X, T, variance=std_dose
    ).get_propensities(X, T)

    clip_value = 1e5

    w_pi = (
        policy_pi.get_propensities(X, T)[:, np.newaxis]
        / estimate_logging_propensities[:, np.newaxis]
    )
    w_pi = np.clip(w_pi, 0, clip_value)

    w_pi_prime = (
        policy_pi_prime.get_propensities(X, T)[:, np.newaxis]
        / estimate_logging_propensities[:, np.newaxis]
    )
    w_pi_prime = np.clip(w_pi_prime, 0, clip_value)

    pi_samples = policy_pi.sample_treatments(X)
    pi_prime_samples = policy_pi_prime.sample_treatments(X)

    return dict(
        X=X,
        T=T,
        Y=Y,
        w_pi=w_pi,
        w_pi_prime=w_pi_prime,
        pi_samples=pi_samples,
        pi_prime_samples=pi_prime_samples,
    )


def make_scenario(scenario_id, seed=None):
    """
    Returns:
        - X: covariates
        - policy_logging: logging policy π₀
        - policy_pi: target policy π
        - policy_pi_prime: alternative policy π′
        - beta, gamma: for outcome model
    """
    if seed is not None:
        np.random.seed(seed)

    reg = LinearRegression().fit(X, T)
    w_base = reg.coef_ + np.random.normal(scale=0.1, size=reg.coef_.shape[0])
    d = reg.coef_.shape[0]

    if scenario_id == "I":
        # Null scenario: π = π′ (no difference)
        policy_pi = GaussianPolicy(w_base, scale=std_dose)
        policy_pi_prime = GaussianPolicy(w_base, scale=std_dose)

    elif scenario_id == "II":
        # Mean shift: π′ mean shifted along one direction
        shift = 2 * np.ones(d)
        policy_pi = GaussianPolicy(w_base, scale=std_dose)
        policy_pi_prime = GaussianPolicy(w_base + shift, scale=std_dose)

    elif scenario_id == "III":
        # Mixture policy π′: bimodal
        w1 = w_base + np.std(w_base) * np.ones(d)
        w2 = w_base - np.std(w_base) * np.ones(d)

        class MixturePolicy:
            def __init__(self, w1, w2):
                self.p1 = GaussianPolicy(w1, scale=std_dose)
                self.p2 = GaussianPolicy(w2, scale=std_dose)

            def sample_treatments(self, X):
                mask = np.random.binomial(1, 0.5, size=X.shape[0])
                T1 = self.p1.sample_treatments(X)
                T2 = self.p2.sample_treatments(X)
                return mask * T1 + (1 - mask) * T2

            def get_propensities(self, X, t):
                return 0.5 * self.p1.get_propensities(
                    X, t
                ) + 0.5 * self.p2.get_propensities(X, t)

        policy_pi = GaussianPolicy(w_base)
        policy_pi_prime = MixturePolicy(w1, w2)

    elif scenario_id == "IV":

        w1 = w_base + 2 * np.std(w_base) * np.ones(d)
        w2 = w_base

        class MixturePolicy:
            def __init__(self, w1, w2):
                self.p1 = GaussianPolicy(w1, scale=std_dose)
                self.p2 = GaussianPolicy(w2, scale=std_dose)

            def sample_treatments(self, X):
                mask = np.random.binomial(1, 0.5, size=X.shape[0])
                T1 = self.p1.sample_treatments(X)
                T2 = self.p2.sample_treatments(X)
                return mask * T1 + (1 - mask) * T2

            def get_propensities(self, X, t):
                return 0.5 * self.p1.get_propensities(
                    X, t
                ) + 0.5 * self.p2.get_propensities(X, t)

        policy_pi = GaussianPolicy(w_base, scale=std_dose)
        policy_pi_prime = MixturePolicy(w1, w2)

    else:
        raise ValueError(f"Unknown scenario {scenario_id}")

    return X, T, Y, policy_pi, policy_pi_prime


def find_best_params(
    X_log, A_log, Y_log, reg_grid=[1e1, 1e0, 0.1, 1e-2, 1e-3, 1e-4], num_cv=3
):
    kr = GridSearchCV(
        KernelRidge(kernel="rbf", gamma=0.1),
        cv=num_cv,
        param_grid={"alpha": reg_grid},
    )
    features = np.concatenate([X_log, A_log.reshape(-1, 1)], axis=1)
    kr.fit(features, Y_log)
    reg_param = kr.best_params_["alpha"]
    return reg_param


# %%
