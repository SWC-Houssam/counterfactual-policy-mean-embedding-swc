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
    """
    Given a binary image (values in {0,1}), return a scaled version:
        - scaled by `low` with prob 0.5
        - scaled by `high` with prob 0.5

    This changes the variance of pixel intensities without changing the mean.

    Parameters:
        image (np.ndarray): binary image (H x W)
        low (float): lower multiplier
        high (float): higher multiplier
        rng (np.random.RandomState or None): for reproducibility

    Returns:
        np.ndarray: new image with same mean but higher variance
    """
    if rng is None:
        rng = np.random.RandomState()

    scale = rng.choice([low, high])
    return scale * image


# %%
class GaussianPolicy2D:
    def __init__(self, theta=0.0, beta=0.0, sigma=0.1):
        self.theta = theta  # angle for projection
        self.beta = beta  # mean shift
        self.sigma = sigma  # std dev per dimension

    def get_mean(self, U):
        mean1 = U[:, 0] * np.cos(self.theta) + self.beta
        mean2 = U[:, 1] * np.sin(self.theta) + self.beta
        return np.stack([mean1, mean2], axis=1)

    def sample_treatments(self, U):
        mean = self.get_mean(U)
        noise = np.random.normal(
            0, self.sigma, size=mean.shape
        )  # independent noise per dim
        return mean + noise

    def get_propensities(self, U, A):
        mean = self.get_mean(U)
        diff = A - mean
        norm_sq = np.sum(diff**2, axis=1)
        d = A.shape[1]  # should be 2
        return (1 / ((2 * np.pi * self.sigma**2) ** (d / 2))) * np.exp(
            -norm_sq / (2 * self.sigma**2)
        )


def generate_logging_data(n, theta=0.0, beta=0.0, sigma=0.1, seed=0):
    rng = np.random.RandomState(seed)
    U = rng.uniform(0, 1, size=(n, 2))
    logging_policy = GaussianPolicy2D(theta, beta, sigma)
    A = logging_policy.sample_treatments(U)
    return U, A, logging_policy


def generate_outcomes(U, A, get_image_by_position):
    posX = np.clip(((U[:, 0] + 1.5) * 32 / 3).astype(int), 0, 31)
    posY = np.clip(((U[:, 1] + 1.5) * 32 / 3).astype(int), 0, 31)
    Y = np.stack([get_image_by_position(px, py) for px, py in zip(posX, posY)], axis=0)
    return Y


def generate_ope_data(
    U, A, Y, logging_policy, policy_pi, policy_pi_prime, clip_value=1e5
):
    pi_density = policy_pi.get_propensities(U, A)
    pi_prime_density = policy_pi_prime.get_propensities(U, A)
    logging_density = logging_policy.get_propensities(U, A)

    w_pi = np.clip(pi_density / logging_density, 0, clip_value)[:, None]
    w_pi_prime = np.clip(pi_prime_density / logging_density, 0, clip_value)[:, None]

    pi_samples = policy_pi.sample_treatments(U)
    pi_prime_samples = policy_pi_prime.sample_treatments(U)

    return dict(
        U=U,
        A=A,
        Y=Y,
        w_pi=w_pi,
        w_pi_prime=w_pi_prime,
        pi_samples=pi_samples,
        pi_prime_samples=pi_prime_samples,
    )

# %%
def make_scenario(scenario_id, sigma=0.1):
    theta_base = 0.0
    beta_base = 0.0

    if scenario_id == "I":
        pi = GaussianPolicy2D(theta=theta_base, beta=beta_base, sigma=sigma)
        pi_prime = GaussianPolicy2D(theta=theta_base, beta=beta_base, sigma=sigma)
    elif scenario_id == "II":
        pi = GaussianPolicy2D(theta=theta_base, beta=beta_base, sigma=sigma)
        pi_prime = GaussianPolicy2D(
            theta=theta_base, beta=beta_base + 0.25, sigma=sigma
        )
    elif scenario_id == "III":

        class MixturePolicy:
            def __init__(self):
                self.p1 = GaussianPolicy2D(
                    theta=theta_base, beta=beta_base + 0.25, sigma=sigma
                )
                self.p2 = GaussianPolicy2D(
                    theta=theta_base, beta=beta_base - 0.25, sigma=sigma
                )

            def get_propensities(self, U, A):
                return 0.5 * self.p1.get_propensities(
                    U, A
                ) + 0.5 * self.p2.get_propensities(U, A)

            def sample_treatments(self, U):
                mask = np.random.binomial(1, 0.5, size=U.shape[0])
                A1 = self.p1.sample_treatments(U)
                A2 = self.p2.sample_treatments(U)
                return mask[:, None] * A1 + (1 - mask[:, None]) * A2

        pi = GaussianPolicy2D(theta=theta_base, beta=beta_base, sigma=sigma)
        pi_prime = MixturePolicy()
    else:
        raise ValueError(f"Unknown scenario: {scenario_id}")

    return pi, pi_prime

# %%

seed = 42
n = 1000
rng = default_rng(seed)
theta = rng.uniform(0, 2 * np.pi)
beta = rng.uniform(-0.2, 0.2)
imgs, latents_bases = load_dsprite_dataset()

# Sample context
U1 = rng.uniform(0, 1, size=n)
U2 = rng.uniform(0, 1, size=n)

# Sample treatment
eps1 = rng.normal(-1, 0.1, size=n)
eps2 = rng.normal(0, 0.1, size=n)
A1 = U1 * np.cos(theta) + beta + eps1
A2 = U2 * np.sin(theta) + beta + eps2
treatment = np.stack([A1, A2], axis=1)

# Compute image position
posX_id = np.clip(((A1 + 1.5) * 32 / 3).astype(int), 0, 31)
posY_id = np.clip(((A2 + 1.5) * 32 / 3).astype(int), 0, 31)

# Map to image index and load images
idx = image_id(latents_bases, posX_id, posY_id)
outcomes = imgs[idx].reshape(n, -1).astype(np.float32)
# %%
import matplotlib.pyplot as plt

def show_image(image_flat):
    """
    image_flat: numpy array of shape (4096,), flattened 64x64 grayscale image
    """
    image = image_flat.reshape(64, 64)
    plt.imshow(image, cmap="gray")
    plt.axis("off")
    plt.show()

# Example usage (assuming you have a treatment image from your dataset)
# treatment: (n_samples, 4096)
# For example, show the first image:
show_image(0.5*outcomes[594])
# %%

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(A1, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title("Histogram of A₁")
plt.xlabel("A₁")
plt.xlim(-1.5, 1.5)
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(A2, bins=30, alpha=0.7, color='green', edgecolor='black')
plt.title("Histogram of A₂")
plt.xlabel("A₂")
plt.xlim(-1.5, 1.5)
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
# %%
outcomes[594]
np.unique(outcomes[594])
# %%
