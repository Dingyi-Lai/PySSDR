import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense, Flatten, ReLU
# import torch.optim as optim

from time import time
from scipy.stats import poisson, gamma, norm
import multiprocessing as mp
# from sddr import Sddr  # Assuming you have pyssdr installed and configured correctly
import logging
from datetime import datetime
# import torch
logging.basicConfig(level=logging.INFO)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ---------------------------
# Helper Functions
# ---------------------------

def scale_to_range(data, lower=-1, upper=1):
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * (upper - lower) + lower

def build_dnn(input_dim, layer_sizes=[32, 16], activation="relu"):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(layer_sizes[0], activation=activation),
        Dense(layer_sizes[1], activation=activation),
        Dense(1, activation=None)
    ])
    return model

def generate_gp_image(grid_size=28, length_scale=0.2, random_state=None):
    np.random.seed(random_state)
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    coords = np.vstack([X.ravel(), Y.ravel()]).T

    kernel = RBF(length_scale=length_scale)
    gp = GaussianProcessRegressor(kernel=kernel)
    intensity = gp.sample_y(coords, n_samples=1).reshape(grid_size, grid_size)
    return scale_to_range(intensity)

# ---------------------------
# Simulation Functions
# ---------------------------

def generate_linear_effects(n_samples, K, I, alpha, random_state=None):
    np.random.seed(random_state)
    X = np.random.normal(size=(n_samples, I))
    X_scaled = scale_to_range(X, 0, 1)
    linear_effects = np.zeros((n_samples, K))

    for k in range(1, K + 1):
        for i in range(I):
            linear_effects[:, k - 1] += alpha[k][i] * X_scaled[:, i]
    return X_scaled, scale_to_range(linear_effects)

def generate_nonlinear_effects(n_samples, K, J, beta, random_state=None):
    np.random.seed(random_state)
    Z = np.random.uniform(0, 1, size=(n_samples, J))
    nonlinear_effects = np.zeros((n_samples, K))

    for k in range(1, K + 1):
        for j in range(J):
            nonlinear_effects[:, k - 1] += beta[k][j](Z[:, j])
    return Z, scale_to_range(nonlinear_effects)

def generate_unstructured_effects(images, dnn_model, K):
    n_samples = len(images)
    unstructured_effects = np.zeros((n_samples, K))

    for k in range(K):
        unstructured_output = np.random.rand(n_samples)  # Mock prediction
        unstructured_effects[:, k] = scale_to_range(unstructured_output)
    return unstructured_effects

def combine_effects(unstructured_effects, linear_effects, nonlinear_effects, SNR):
    K = linear_effects.shape[1]
    N = linear_effects.shape[0]  # Number of data points
    etas = np.zeros((N, K))
    a = np.zeros((N, K))  # Intercepts for each data point and K

    for k in range(K):
        etas[:, k] = linear_effects[:, k] + nonlinear_effects[:, k] + unstructured_effects[:, k]
        range_etas = np.ptp(etas[:, k])
        a[:, k] = np.random.normal(0, range_etas/SNR, size=N)  # Gaussian noise for each data point
        etas[:, k] += a[:, k]  # Add noise to each data point
    
    return etas, a

def simulate_response(etas, distribution="poisson", K=2):
    n_samples = etas.shape[0]

    if distribution == "poisson" and K == 1:
        mu = np.exp(etas[:, 0])
        return np.random.poisson(mu)
    elif distribution == "gamma" and K == 2:
        mu = np.exp(etas[:, 0])
        sigma = 1.0
        shape = (mu / sigma) ** 2
        scale = sigma ** 2 / mu
        return np.random.gamma(shape, scale, n_samples)
    elif distribution == "gaussian" and K == 2:
        mu = etas[:, 0]
        sigma = np.exp(etas[:, 1])
        return np.random.normal(mu, sigma, n_samples)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

def save_results(scenario, rep, n_samples, X, Z, images, etas, responses, unstructured_effects, intercepts, save_path):
    # save_path = os.path.join(os.environ["TMPDIR"], "output")
    os.makedirs(save_path, exist_ok=True)
    scenario_index = '_'.join(map(str, scenario))
    
    # Save data as before
    np.save(f"{save_path}/X_{scenario_index}_rep{rep}.npy", X)
    np.save(f"{save_path}/Z_{scenario_index}_rep{rep}.npy", Z)
    np.save(f"{save_path}/images_{scenario_index}_rep{rep}.npy", images)
    np.save(f"{save_path}/etas_{scenario_index}_rep{rep}.npy", etas)
    np.save(f"{save_path}/responses_{scenario_index}_rep{rep}.npy", responses)

    # Save intercepts and unstructured effects separately
    np.save(f"{save_path}/intercepts_{scenario_index}_rep{rep}.npy", intercepts)
    np.save(f"{save_path}/unstructured_effects_{scenario_index}_rep{rep}.npy", unstructured_effects)

    logging.info(f"Saved dataset {scenario_index}_rep{rep}, responses, intercepts, and unstructured_effects.")

# ---------------------------
# Generate Task Function for Parallel Execution
# ---------------------------


def generate_task(task_index, scenario_index, rep, scenarios, alpha, beta, grid_size, save_path):
    scenario = scenarios[scenario_index]
    n_samples, distribution, K, SNR = scenario
    
    # Build the DNN model (mock)
    dnn_model = build_dnn(grid_size * grid_size)

    logging.info(f"Generating dataset: Scenario {scenario_index} | Replication {rep} | {scenario}")
    
    # Generate structured and unstructured effects
    X, linear_effects = generate_linear_effects(n_samples, K, 2, alpha, random_state=task_index)
    Z, nonlinear_effects = generate_nonlinear_effects(n_samples, K, 2, beta, random_state=task_index)
    images = np.array([generate_gp_image(grid_size, random_state=task_index * 5000 + i) for i in range(n_samples)])
    unstructured_effects = generate_unstructured_effects(images, dnn_model, K)
    logging.info(f"Generated X, Z, images and unstructured_effects: Scenario {scenario_index} | Replication {rep} | {scenario}")
    
    # Combine effects and calculate intercepts
    etas, intercepts = combine_effects(unstructured_effects, linear_effects, nonlinear_effects, SNR)
    logging.info(f"Generated etas and intercepts: Scenario {scenario_index} | Replication {rep} | {scenario}")
    
    # Generate response based on distribution
    responses = simulate_response(etas, distribution, K)
    logging.info(f"Generated responses: Scenario {scenario_index} | Replication {rep} | {scenario}")
    
    # Save datasets and responses, including intercepts and unstructured effects
    save_results(scenario, rep, n_samples, X, Z, images, etas, responses, unstructured_effects, intercepts, save_path)

# ---------------------------
# Parallel Execution Function
# ---------------------------

def parallel_generate(scenarios, grid_size, alpha, beta, n_cores=4, n_rep=4, save_path=None):
    logging.info("Starting parallel dataset generation...")
    start_time = datetime.now()
    
    # Create (scenario_index, rep) combinations for parallel processing
    task_list = [(i, r) for i in range(len(scenarios)) for r in range(n_rep)]
    logging.info(f"number of task_list: {len(task_list)}.")
    logging.info(f"number of cores used: {n_cores}.")
    
    # Use pool.starmap to pass multiple arguments to the method
    with mp.Pool(n_cores) as pool:
        pool.starmap(generate_task, [(idx, i, r, scenarios, alpha, beta, grid_size, save_path) 
                                     for idx, (i, r) in enumerate(task_list)])
    
    end_time = datetime.now()
    logging.info(f"Parallel dataset generation completed in {end_time - start_time}.")

# ---------------------------
# Scenario Setup
# ---------------------------

scenarios = [
    (100, "poisson", 1, 2) #, (500, "poisson", 1, 2), (1000, "poisson", 1, 2),
#     (100, "poisson", 1, 4), (500, "poisson", 1, 4), (1000, "poisson", 1, 4),
#     (100, "gamma", 2, 2), (500, "gamma", 2, 2), (1000, "gamma", 2, 2), 
#    (100, "gamma", 2, 4), (500, "gamma", 2, 4), (1000, "gamma", 2, 4), 
#    (100, "gaussian", 2, 2), (500, "gaussian", 2, 2), (1000, "gaussian", 2, 2),
#    (100, "gaussian", 2, 4), (500, "gaussian", 2, 4), (1000, "gaussian", 2, 4)
]

grid_size = 28

# Define coefficients
alpha = {1: [3, -1], 2: [-0.5, 6]}

# Define named functions for nonlinear effects
def nonlinear_effect_1_z1(z1):
    return scale_to_range(3 * np.sin(6 * z1))

def nonlinear_effect_1_z2(z2):
    return scale_to_range(np.exp(5 * z2))

def nonlinear_effect_2_z1(z1):
    return scale_to_range(np.cos(8 * z1))

def nonlinear_effect_2_z2(z2):
    return scale_to_range(0.1 * np.sqrt(z2))

# Example coefficients for nonlinear effects
beta = {
    1: [nonlinear_effect_1_z1, nonlinear_effect_1_z2],
    2: [nonlinear_effect_2_z1, nonlinear_effect_2_z2]
}
# save_path = os.path.join(os.environ["TMPDIR"], "output")
save_path = os.path.join("output_local")
# parallel_generate(scenarios, grid_size, alpha, beta, n_cores=152, n_rep=100, save_path=save_path)
parallel_generate(scenarios, grid_size, alpha, beta, n_cores=4, n_rep=100, save_path=save_path)
