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
from concurrent.futures import ThreadPoolExecutor

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
    model.compile(optimizer='adam', loss='mse')  # Compile the model

    return model

def generate_gp_image(grid_size=28, length_scale=0.2, random_state=None): # , SNR_compare=[2,4]
    np.random.seed(random_state)
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    coords = np.vstack([X.ravel(), Y.ravel()]).T

    kernel = RBF(length_scale=length_scale)
    gp = GaussianProcessRegressor(kernel=kernel, random_state=random_state)
    intensity = gp.sample_y(coords, n_samples=1, random_state=random_state).reshape(grid_size, grid_size)
    
    intensity_scaled = scale_to_range(intensity)
    
    return intensity_scaled

# ---------------------------
# Simulation Functions
# ---------------------------

def generate_linear_effects(n_samples, I, K, alpha, random_state=None): # SNR_compare, 
    np.random.seed(random_state)
    X = np.random.normal(size=(I, n_samples, K)) # The same basis for all SNRs
    X_scaled = np.zeros_like(X)
    linear_effects1 = np.zeros((I, n_samples, K))
    
    for i in range(I):
        X_scaled[i] = scale_to_range(X[i], 0, 1)
        for k in range(K):
            linear_effects1[i, :, k] = alpha[k][i] * X_scaled[i][:, k]
        
    return X_scaled, scale_to_range(linear_effects1)

def generate_nonlinear_effects(n_samples, J, K, beta, random_state=None): #SNR_compare, 
    np.random.seed(random_state)
    Z = np.random.uniform(0, 1, size=(J, n_samples, K)) # doesn't need to scale
    nonlinear_effects1 = np.zeros((J, n_samples, K))

    for j in range(J):
        for k in range(K):
            nonlinear_effects1[j, :, k] = beta[k][j](Z[j][:, k])
        
    return Z, scale_to_range(nonlinear_effects1) #a_Z1, a_Z2, , scale_to_range(nonlinear_effects2)

def generate_unstructured_effects(images, dnn_model, K):
    n_samples = len(images)
    unstructured_effects = np.zeros((n_samples, K))
    
    # Predict unstructured effects using the DNN model
    predictions = dnn_model.predict(images)

    # Extract the penultimate layer's output
    penultimate_layer_model = Model(inputs=dnn_model.input, outputs=dnn_model.layers[-2].output)
    penultimate_output = penultimate_layer_model.predict(images)

    # Extract the weights and bias from the final layer
    final_layer_weights, final_layer_bias = dnn_model.layers[-1].get_weights()

    for k in range(K):
        unstructured_effects[:, k] = scale_to_range(predictions[:, 0])

    return unstructured_effects, penultimate_output, final_layer_weights, final_layer_bias # \hat{U}_k, \hat{\psi}_k, \hat{b}_k

def combine_effects(scenario, rep, save_path, unstructured_effects, linear_effects, nonlinear_effects, distribution="poisson", SNR_list=[1,8]):
    for s in SNR_list:
        add_SNR(scenario, rep, save_path, unstructured_effects, linear_effects, nonlinear_effects, distribution, s)

def add_SNR(scenario_index, rep, save_path, unstructured_effects, linear_effects, nonlinear_effects, distribution="poisson", SNR=1):
    os.makedirs(save_path, exist_ok=True)
    scenario_index += f"_SNR{SNR}_rep{rep}"
    K = linear_effects.shape[2]
    N = linear_effects.shape[1]  # Number of data points
    etas = np.zeros((N, K))
    k = 0
    etas[:, k] = linear_effects[:, :, k].sum(axis=0) + nonlinear_effects[:, :, k].sum(axis=0) + unstructured_effects[:, k]
    range_etas = np.ptp(etas[:, k])
    std_eta = np.std(etas[:, k])
    if distribution == "poisson" or distribution == "gamma":
        a = std_eta * SNR - range_etas
        etas[:, k] += a
    elif distribution == "gaussian":
        for k in range(K):
            etas[:, k] = linear_effects[:, :, k].sum(axis=0) + nonlinear_effects[:, :, k].sum(axis=0) + unstructured_effects[:, k]
        a = np.exp(etas[:, 1]) * SNR - np.ptp(etas[:, 0]) # a2 in (17)
        etas[:, 1] += a
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    
    
    save_with_var_name(a, 'a', 'npy', save_path, scenario_index, rep)
    save_with_var_name(etas, 'etas', 'npy', save_path, scenario_index, rep)
    # Generate response based on distribution
    responses = simulate_response(etas, distribution)
    save_with_var_name(responses, 'responses', 'npy', save_path, scenario_index, rep)

def simulate_response(etas, distribution="poisson"):
    n_samples = etas.shape[0]
    if distribution == "poisson":
        mu = np.exp(etas[:, 0])
        return np.random.poisson(mu)
    elif distribution == "gamma":
        mu = np.exp(etas[:, 0])
        sigma = 1.0
        shape = (mu / sigma) ** 2
        scale = sigma ** 2 / mu
        return np.random.gamma(shape, scale, n_samples) # scale must be non-negative
    elif distribution == "gaussian":
        mu = etas[:, 0]
        sigma = np.exp(etas[:, 1])
        return np.random.normal(mu, sigma, n_samples)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

def save_with_var_name(var, var_name, var_type, save_path, scenario_index, rep):
    if var_type == 'npy':
        np.save(f"{save_path}/{var_name}_{scenario_index}.npy", var)
    if var_type == 'keras':
        var.save(f"{save_path}/{var_name}_{scenario_index}.keras")
    logging.info(f"Saved {var_name} to {save_path}/{var_name}_{scenario_index}.npy")
    
# ---------------------------
# Generate Task Function for Parallel Execution
# ---------------------------

def generate_task(n_sample, distribution_list, SNR_list, grid_size, alpha_l, beta_nl, n_rep, 
                  save_path, compute_type='parallel'):
    
    # Set random seed for reproducibility
    np.random.seed(n_sample+n_rep)
    # Build the DNN model (mock)
    dnn_model = build_dnn(grid_size * grid_size)

    logging.info(f"Generating dataset: Number of obs {n_sample} | Replication {n_rep}")
    I = 2
    K = 2 # predefine
    # Generate structured and unstructured effects
    X, linear_effects = generate_linear_effects(n_sample, I, K, alpha_l, random_state=n_sample+n_rep)
    Z, nonlinear_effects = generate_nonlinear_effects(n_sample, I, K, beta_nl, random_state=n_sample+n_rep)
    images = np.zeros((n_sample, grid_size, grid_size))

    for i in range(n_sample):
        images[i] = generate_gp_image(grid_size, length_scale=0.2, random_state=(n_sample+n_rep)*n_sample-i)
    
    # Flatten images before passing to the DNN model
    flattened_images = images.reshape(n_sample, -1)  
    unstructured_effects, U_k, psi_k, b_k = generate_unstructured_effects(flattened_images, dnn_model, K)

    logging.info(f"Generated X, Z, images and unstructured_effects: Number of obs {n_sample} | Replication {n_rep}")
    scenario_index = '_'.join(map(str, ['n', n_sample, 'rep',n_rep]))

    save_with_var_name(X, 'X', 'npy', save_path, scenario_index, n_rep)
    save_with_var_name(linear_effects, 'linear_effects', 'npy', save_path, scenario_index, n_rep)
    save_with_var_name(Z, 'Z', 'npy', save_path, scenario_index, n_rep)
    save_with_var_name(nonlinear_effects, 'nonlinear_effects', 'npy', save_path, scenario_index, n_rep)
    save_with_var_name(images, 'images', 'npy', save_path, scenario_index, n_rep)
    save_with_var_name(unstructured_effects, 'unstructured_effects', 'npy', save_path, scenario_index, n_rep)
    save_with_var_name(U_k, 'U_k', 'npy', save_path, scenario_index, n_rep)
    save_with_var_name(psi_k, 'psi_k', 'npy', save_path, scenario_index, n_rep)
    save_with_var_name(b_k, 'b_k', 'npy', save_path, scenario_index, n_rep)
    save_with_var_name(dnn_model, 'dnn_model', 'keras', save_path, scenario_index, n_rep)
    
    # Combine effects
    # Parallel processing
    if compute_type == 'parallel':
        # parellel computing can't be nested by default, use concurrent computing for each executer
        with ThreadPoolExecutor() as inner_executor:
            inner_executor.map(combine_effects, [(scenario_index, n_rep, save_path, unstructured_effects, linear_effects, nonlinear_effects, d, SNR_list) 
                                        for d in distribution_list])
    if compute_type == 'serial':
        for d in distribution_list:
            print(d)
            combine_effects(scenario_index, n_rep, save_path, unstructured_effects, linear_effects, nonlinear_effects, d, SNR_list)    
    
# ---------------------------
# Parallel Execution Function
# ---------------------------

def scenarios_generate(n_list, distribution_list, SNR_list, grid_size, alpha_l, beta_nl, n_rep, n_cores,
                  compute_type='parallel'):
    logging.info(f"Starting {compute_type} dataset generation...")
    start_time = datetime.now()
    n_r = [(i, r) for i in n_list for r in range(n_rep)]
    if compute_type == 'parallel':
        # Use pool.starmap to pass multiple arguments to the method
        save_path = os.path.join(os.environ["TMPDIR"], "output")
        with mp.Pool(n_cores) as pool:
            pool.starmap(generate_task, [(i, distribution_list, SNR_list, grid_size, 
                                            alpha_l, beta_nl, r, save_path, compute_type) 
                                        for (i, r) in n_r])
    if compute_type == 'serial':
        save_path = "../data_generation/output_local"
        for (i, r) in n_r:
            generate_task(n_sample=i, distribution_list=distribution_list, 
                        SNR_list=SNR_list, grid_size=grid_size, alpha_l=alpha_l,
                        beta_nl=beta_nl, n_rep=r, 
                        save_path=save_path, compute_type=compute_type)
    end_time = datetime.now()
    logging.info(f"Parallel dataset generation completed in {end_time - start_time}.")

# ---------------------------
# Scenario Setup
# ---------------------------

# n_list = [100, 500, 1000]
n_list = [10]
distribution_list = ["poisson", "gamma", "gaussian"]
SNR_list=[1,8]

grid_size = 28
n_core = mp.cpu_count()
print(f"Number of cores: {n_core}")
# Define coefficients
alpha_l = {0: [3, -1], 1: [-0.5, 6]}

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
beta_nl = {
    0: [nonlinear_effect_1_z1, nonlinear_effect_1_z2],
    1: [nonlinear_effect_2_z1, nonlinear_effect_2_z2]
}

scenarios_generate(n_list, distribution_list, SNR_list, grid_size, alpha_l, beta_nl, n_rep=1, n_cores=n_core,
                  compute_type='parallel')