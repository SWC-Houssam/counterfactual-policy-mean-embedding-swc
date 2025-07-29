# %%
import numpy as np
from numpy.random import default_rng
from filelock import FileLock
import pathlib

# %%
DATA_PATH = pathlib.Path(__file__).resolve().parent.parent.joinpath("testing/dsprite")


def load_dsprite_dataset():
    with FileLock("./data.lock"):
        dataset_zip = np.load(
            DATA_PATH.joinpath("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"),
            allow_pickle=True,
            encoding="bytes",
        )
    imgs = dataset_zip["imgs"]
    metadata = dataset_zip["metadata"][()]
    latents_sizes = metadata[b"latents_sizes"]
    latents_bases = np.concatenate(
        (latents_sizes[::-1].cumprod()[::-1][1:], np.array([1]))
    )
    return imgs, latents_bases


def image_id(latent_bases, posX_id_arr, posY_id_arr):
    data_size = posX_id_arr.shape[0]
    color_id_arr = np.zeros(data_size, dtype=int)
    shape_id_arr = np.full(data_size, 2, dtype=int)
    scale_id_arr = np.zeros(data_size, dtype=int)
    orientation_id_arr = np.zeros(data_size, dtype=int)
    indices = np.stack(
        [
            color_id_arr,
            shape_id_arr,
            scale_id_arr,
            orientation_id_arr,
            posX_id_arr,
            posY_id_arr,
        ],
        axis=1,
    )
    return indices.dot(latent_bases)


def structured_outcome_with_mixture_contrast(image: np.ndarray, low=0.5, high=1.5, rng=None) -> np.ndarray:
    if rng is None:
        rng = np.random.RandomState()
    scale = rng.choice([low, high])
    return scale * image


class GaussianPolicy2D:
    def __init__(self, theta=0.0, beta=0.0, sigma=0.1):
        self.theta = theta
        self.beta = beta
        self.sigma = sigma

    def get_mean(self, U):
        mean1 = U[:, 0] * np.cos(self.theta) + self.beta
        mean2 = U[:, 1] * np.sin(self.theta) + self.beta
        return np.stack([mean1, mean2], axis=1)

    def sample_treatments(self, U):
        mean = self.get_mean(U)
        noise = np.random.normal(0, self.sigma, size=mean.shape)
        return mean + noise

    def get_propensities(self, U, A):
        mean = self.get_mean(U)
        diff = A - mean
        norm_sq = np.sum(diff**2, axis=1)
        d = A.shape[1]
        return (1 / ((2 * np.pi * self.sigma**2) ** (d / 2))) * np.exp(
            -norm_sq / (2 * self.sigma**2)
        )


def make_scenario(scenario_id, sigma=0.1):
    theta_base = 0.0
    beta_base = 0.0

    if scenario_id == "I":
        pi = GaussianPolicy2D(theta=theta_base, beta=beta_base, sigma=sigma)
        pi_prime = GaussianPolicy2D(theta=theta_base, beta=beta_base, sigma=sigma)

    elif scenario_id == "III":
        # Same policy but outcomes modified via contrast scaling
        pi = GaussianPolicy2D(theta=theta_base, beta=beta_base, sigma=sigma)
        pi_prime = pi  # same action policy, outcome differs

    elif scenario_id == "IV":
        pi = GaussianPolicy2D(theta=theta_base, beta=beta_base + 0.25, sigma=sigma)
        pi_prime = GaussianPolicy2D(theta=theta_base, beta=beta_base - 0.25, sigma=sigma)

    else:
        raise ValueError(f"Unknown scenario: {scenario_id}")

    return pi, pi_prime


def generate_logging_data(n, theta=0.0, beta=0.0, sigma=0.1, seed=0):
    rng = np.random.RandomState(seed)
    U = rng.uniform(0, 1, size=(n, 2))
    logging_policy = GaussianPolicy2D(theta, beta, sigma)
    A = logging_policy.sample_treatments(U)
    return U, A, logging_policy


def generate_outcomes(U, A, imgs, latents_bases, rng=None, scenario=None):
    posX = np.clip(((U[:, 0] + 1.5) * 32 / 3).astype(int), 0, 31)
    posY = np.clip(((U[:, 1] + 1.5) * 32 / 3).astype(int), 0, 31)
    image_indices = image_id(latents_bases, posX, posY)

    if rng is None:
        rng = np.random.RandomState()

    Y = []
    for idx in image_indices:
        img = imgs[idx].astype(np.float32)
        if scenario == "III":
            img = structured_outcome_with_mixture_contrast(img, rng=rng)
        Y.append(img)
    return np.stack(Y, axis=0)


def generate_ope_data(U, A, Y, logging_policy, pi, pi_prime, clip_value=1e5):
    pi_density = pi.get_propensities(U, A)
    pi_prime_density = pi_prime.get_propensities(U, A)
    logging_density = logging_policy.get_propensities(U, A)

    w_pi = np.clip(pi_density / logging_density, 0, clip_value)[:, None]
    w_pi_prime = np.clip(pi_prime_density / logging_density, 0, clip_value)[:, None]

    pi_samples = pi.sample_treatments(U)
    pi_prime_samples = pi_prime.sample_treatments(U)

    return dict(
        U=U,
        A=A,
        Y=Y,
        w_pi=w_pi,
        w_pi_prime=w_pi_prime,
        pi_samples=pi_samples,
        pi_prime_samples=pi_prime_samples,
    )


def run_dsprite_experiment(scenario_id="I", seed=42, n=1000, sigma=0.1):
    rng = np.random.RandomState(seed)
    imgs, latents_bases = load_dsprite_dataset()
    pi, pi_prime = make_scenario(scenario_id, sigma)
    U, A, logging_policy = generate_logging_data(n=n, sigma=sigma, seed=seed)
    Y = generate_outcomes(U, A, imgs, latents_bases, rng=rng, scenario=scenario_id)
    data = generate_ope_data(U, A, Y, logging_policy, pi, pi_prime)
    return data
