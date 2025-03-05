import numpy as np
from PIL import Image
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import tensorflow as tf
from tensorflow import keras
from keras import Input, Model
from keras.models import load_model, Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

from time import time
from scipy.stats import poisson, gamma, norm
from scipy.optimize import minimize_scalar
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import logging
from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

logging.basicConfig(level=logging.INFO)


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

def generate_gp_image(grid_size=28, length_scale=0.2, random_state=None):
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

def generate_linear_effects(n_samples, I, K, alpha, random_state=None):
    """
    Generates linear effects for X1, X2, etc.
    alpha[k][i] is the coefficient for X_i in the k-th dimension of eta.
    """
    np.random.seed(random_state)
    # X has shape (I, n_samples)
    X = np.random.normal(size=(I, n_samples))
    X_scaled = np.zeros_like(X)
    linear_effects = np.zeros((I, n_samples, K))
    
    for i in range(I):
        X_scaled[i] = scale_to_range(X[i], -1, 1)
        for k in range(K):
            linear_effects[i, :, k] = alpha[k][i] * X_scaled[i]
        
    return X_scaled, linear_effects # scale_to_range(linear_effects)

def generate_nonlinear_effects(n_samples, J, K, beta, random_state=None):
    """
    Generates nonlinear effects for Z1, Z2, etc.
    beta[k][j] is a function, e.g. beta[0][0] = f(Z1).
    """
    np.random.seed(random_state)
    Z = np.random.uniform(-1, 1, size=(J, n_samples))  # shape (J, n_samples)
    nonlinear_effects = np.zeros((J, n_samples, K))

    for j in range(J):
        for k in range(K):
            nonlinear_effects[j, :, k] = beta[k][j](Z[j])
        
    return Z, scale_to_range(nonlinear_effects)

def generate_unstructured_effects(images, save_path, scenario_index, K):
    """
    Generates unstructured effects by passing images through a small DNN.
    """
    n_samples = len(images)
    unstructured_effects = np.zeros((n_samples, K))
    
    # Build and train a small DNN (mock training here).
    grid_size = images.shape[1]  # e.g., 28 if images are 28x28
    dnn_model = build_dnn(grid_size * grid_size)
    
    # Mock training: We won't actually train, but let's do a quick .fit so the weights update a bit.
    # For demonstration, we just pass random data as 'y'.
    dnn_model.fit(images, np.random.rand(n_samples, 1), epochs=1, verbose=0)
    
    # Predict unstructured effects using the DNN model
    predictions = dnn_model.predict(images)
    
    # Extract the penultimate layer's output
    penultimate_layer_model = Model(inputs=dnn_model.inputs, outputs=dnn_model.layers[-2].output)
    penultimate_output = penultimate_layer_model.predict(images)

    # Extract the weights and bias from the final layer
    final_layer_weights, final_layer_bias = dnn_model.layers[-1].get_weights()

    # For simplicity, just scale predictions for each dimension k
    for k in range(K):
        unstructured_effects[:, k] = scale_to_range(predictions[:, 0])

    save_with_var_name(dnn_model, 'dnn_model', 'keras', save_path, scenario_index)
    return unstructured_effects, penultimate_output, final_layer_weights, final_layer_bias

def combine_effects(scenario_index, save_path,
                    unstructured_effects,
                    linear_effects,
                    nonlinear_effects,
                    distribution="poisson",
                    s=1,
                    add_linear=True,
                    add_nonlinear=False,
                    add_unstructured=False):
    """
    Combines the various effect components into etas and simulates responses.
    """
    scenario_index += f"_dist_{distribution}_SNR_{s}"
    K = linear_effects.shape[2]
    N = linear_effects.shape[1]  # Number of data points
    
    # Initialize etas: shape (N, K)
    etas = np.zeros((N, K))

    # For the first dimension (often 'location' or 'mean'):
    # Conditionally add linear
    if add_linear:
        etas[:, 0] += linear_effects[:, :, 0].sum(axis=0)
    # Conditionally add nonlinear
    if add_nonlinear:
        etas[:, 0] += nonlinear_effects[:, :, 0].sum(axis=0)
    # Conditionally add unstructured
    if add_unstructured:
        etas[:, 0] += unstructured_effects[:, 0]

    if distribution == "poisson":
        # Adjust for SNR
        a = find_a_for_target_snr(s, etas, "poisson")
        etas[:, 0] += a

    elif distribution == "gaussian_homo":
        # For homoscedastic Gaussian, we only have one dimension for mu in etas[:,0].
        # The second dimension is constant log-sigma => put it in etas[:,1].
        a = find_a_for_target_snr(s, etas, "gaussian_homo")
        etas[:, 1] = a

    elif distribution == "gaussian_hetero":
        # For heteroscedastic Gaussian, we also have linear+nonlinear+unstructured for scale
        # in the second dimension (index 1).
        if add_linear:
            etas[:, 1] += linear_effects[:, :, 1].sum(axis=0)
        if add_nonlinear:
            etas[:, 1] += nonlinear_effects[:, :, 1].sum(axis=0)
        if add_unstructured:
            etas[:, 1] += unstructured_effects[:, 1]

        a = find_a_for_target_snr(s, etas, "gaussian_hetero")
        etas[:, 1] += a

    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    
    # Save offset a, etas, then simulate y
    save_with_var_name(a, 'a', 'npy', save_path, scenario_index)
    save_with_var_name(etas, 'etas', 'npy', save_path, scenario_index)
    y = simulate_response(etas, distribution, a)
    save_with_var_name(y, 'y', 'npy', save_path, scenario_index)

    return "done"

def combine_effects_wrapper(args):
    return combine_effects(*args)

def compute_snr(a, etas, dist):
    if dist == "poisson":
        lambda_vals = np.exp(etas[:, 0] + a)
        range_log_lambda = np.ptp(etas[:, 0] + a)
        mean_sqrt_lambda = np.sqrt(lambda_vals)
        return (range_log_lambda / mean_sqrt_lambda).mean()
    if dist == "gaussian_homo":
        sigma = a
        range_mu = np.ptp(etas[:, 0])
        return (range_mu / sigma).mean()
    if dist == "gaussian_hetero":
        sigma = np.exp(etas[:, 1] + a)
        range_mu = np.ptp(etas[:, 0])
        return (range_mu / sigma).mean()

def find_a_for_target_snr(target_snr, etas, dist):
    def loss_function(a):
        computed_snr = compute_snr(a, etas, dist)
        return (computed_snr - target_snr)**2
    
    result = minimize_scalar(loss_function, bounds=(-10, 10), method='bounded')
    return result.x    

def simulate_response(etas, distribution, a):
    """
    Simulates final response y given etas and distribution.
    """
    n_samples = etas.shape[0]
    if "poisson" in distribution:
        mu = np.exp(etas[:, 0])
        return np.random.poisson(mu)
    elif "gamma" in distribution:
        mu = np.exp(etas[:, 0]) # non-negative
        sigma = a
        shape = (mu / sigma) ** 2
        scale = sigma ** 2 / mu
        return np.random.gamma(shape, scale, n_samples) # scale must be non-negative
    elif "gaussian" in distribution:
        mu = etas[:, 0]
        sigma = np.exp(etas[:, 1])  # for hetero; if homo, etas[:,1] = const log-sigma
        return np.random.normal(mu, sigma, n_samples)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

def save_with_var_name(var, var_name, var_type, save_path, scenario_index):
    """
    Helper to save various file types (npy, keras, jpgs).
    """
    if var_type == 'npy':
        np.save(f"{save_path}/{var_name}_{scenario_index}.npy", var)
    if var_type == 'keras':
        var.save(f"{save_path}/{var_name}_{scenario_index}.keras")
    if var_type == 'jpgs':
        images_path = f"{save_path}/{var_name}_{scenario_index}"
        os.makedirs(images_path, exist_ok=True)
        for idx, img in enumerate(var):
            normalized_img = (img * 255 / np.max(img)).astype(np.uint8)
            normalized_img = Image.fromarray(normalized_img).convert("L")
            normalized_img.save(f"{images_path}/{var_name}_{scenario_index}_{idx}.jpg")
    if var_type == 'npz':
        np.savez_compressed(f"{save_path}/{var_name}_{scenario_index}.npz", **var)
    logging.info(f"Saved {var_name} to {save_path}/{var_name}_{scenario_index}.npy")

def read_with_var_name(var_name, var_type, save_path, scenario_index):
    if var_type == 'npy':
        return np.load(f"{save_path}/{var_name}_{scenario_index}.npy")
    if var_type == 'keras':
        return tf.keras.models.load_model(f"{save_path}/{var_name}_{scenario_index}.keras")


# ---------------------------
# Main Task Generation
# ---------------------------

def generate_task(n_sample, distribution_list, SNR_list, grid_size, alpha_l, beta_nl, n_rep, 
                  save_path, compute_type='parallel',
                  add_linear=True,
                  add_nonlinear=False,
                  add_unstructured=False):
    """
    Generates one scenario (for a given n_sample and replication index),
    optionally including nonlinear and unstructured effects.
    """
    np.random.seed(n_sample + n_rep)
    logging.info(f"Generating dataset: Number of obs {n_sample} | Replication {n_rep}")

    I = 2
    K = 2  # location + scale dimension
    scenario_index = '_'.join(map(str, ['n', n_sample, 'rep', n_rep]))
    # 1) Generate linear effects
    if add_linear:
        X, linear_effects = generate_linear_effects(n_sample, I, K, alpha_l, random_state=n_sample+n_rep)
        save_with_var_name(X, 'X', 'npy', save_path, scenario_index)
        save_with_var_name(linear_effects, 'linear_effects', 'npy', save_path, scenario_index)
    else:
        # Create placeholders (zeros) so it contributes nothing to eta
        X = np.zeros((I, n_sample))
        linear_effects = np.zeros((I, n_sample, K))
    # 2) Generate nonlinear effects (only if add_nonlinear=True)
    if add_nonlinear:
        Z, nonlinear_effects = generate_nonlinear_effects(n_sample, I, K, beta_nl, random_state=n_sample+n_rep)
        save_with_var_name(Z, 'Z', 'npy', save_path, scenario_index)
        save_with_var_name(nonlinear_effects, 'nonlinear_effects', 'npy', save_path, scenario_index)
    else:
        # Create placeholders (zeros) so it contributes nothing to eta
        Z = np.zeros((I, n_sample))
        nonlinear_effects = np.zeros((I, n_sample, K))

    # 3) Generate images if unstructured is True
    if add_unstructured:
        images = np.zeros((n_sample, grid_size, grid_size))
        for i in range(n_sample):
            images[i] = generate_gp_image(grid_size, length_scale=0.2,
                                          random_state=(n_sample + n_rep)*n_sample - i)
        save_with_var_name(images, 'images', 'npy', save_path, scenario_index)
        save_with_var_name(images, 'images_jpg', 'jpgs', save_path, scenario_index)
    else:
        images = None

    logging.info(f"Generated X, Z, images: n={n_sample}, rep={n_rep}")

    # 5) Flatten images and generate unstructured effects if add_unstructured=True
    if add_unstructured:
        flattened_images = images.reshape(n_sample, -1)
        unstructured_effects, U_k, psi_k, b_k = generate_unstructured_effects(
            flattened_images, save_path, scenario_index, K
        )
        save_with_var_name(unstructured_effects, 'unstructured_effects', 'npy', save_path, scenario_index)
        save_with_var_name(U_k, 'U_k', 'npy', save_path, scenario_index)
        save_with_var_name(psi_k, 'psi_k', 'npy', save_path, scenario_index)
        save_with_var_name(b_k, 'b_k', 'npy', save_path, scenario_index)
    else:
        unstructured_effects = np.zeros((n_sample, K))

    # 6) Combine effects (for each distribution & SNR)
    if compute_type == 'parallel':
        # parallel computing can't be nested by default, so use ThreadPoolExecutor
        with ThreadPoolExecutor() as inner_executor:
            result = list(inner_executor.map(
                combine_effects_wrapper,
                [
                    (
                        scenario_index,
                        save_path,
                        unstructured_effects,
                        linear_effects,
                        nonlinear_effects,
                        d,
                        s,
                        add_linear,
                        add_nonlinear,
                        add_unstructured
                    )
                    for s in SNR_list for d in distribution_list
                ]
            ))
    else:  # compute_type == 'serial'
        for d in distribution_list:
            for s in SNR_list:
                combine_effects(
                    scenario_index,
                    save_path,
                    unstructured_effects,
                    linear_effects,
                    nonlinear_effects,
                    distribution=d,
                    s=s,
                    add_linear=add_linear,
                    add_nonlinear=add_nonlinear,
                    add_unstructured=add_unstructured
                )

    logging.info(f"Generated responses: Number of obs={n_sample}, Rep={n_rep}")

# ---------------------------
# Parallel Execution Function
# ---------------------------

def scenarios_generate(n_list,
                       distribution_list,
                       SNR_list,
                       grid_size,
                       alpha_l,
                       beta_nl,
                       n_rep,
                       n_cores,
                       compute_type='parallel',
                       add_linear=True,
                       add_nonlinear=False,
                       add_unstructured=False):
    """
    Main entry point to generate multiple scenarios in either parallel or serial.
    """
    logging.info(f"Starting {compute_type} dataset generation...")
    start_time = datetime.now()

    # Decide on the folder to store data
    if compute_type == 'parallel':
        save_path = os.path.join(os.environ["TMPDIR"], "output_structured[-1_1]")  # or another path
    else:
        save_path = "../data_generation/output_structured[-1_1]"
    os.makedirs(save_path, exist_ok=True)

    # Prepare (n, rep) tasks
    n_r = [(i, r) for i in n_list for r in range(n_rep)]

    if compute_type == 'parallel':
        with mp.Pool(n_cores) as pool:
            pool.starmap(
                generate_task,
                [
                    (
                        i,
                        distribution_list,
                        SNR_list,
                        grid_size,
                        alpha_l,
                        beta_nl,
                        r,
                        save_path,
                        compute_type,
                        add_linear,
                        add_nonlinear,
                        add_unstructured
                    )
                    for (i, r) in n_r
                ]
            )
    else:  # compute_type == 'serial'
        for (i, r) in n_r:
            generate_task(
                n_sample=i,
                distribution_list=distribution_list,
                SNR_list=SNR_list,
                grid_size=grid_size,
                alpha_l=alpha_l,
                beta_nl=beta_nl,
                n_rep=r,
                save_path=save_path,
                compute_type=compute_type,
                add_linear=add_linear,
                add_nonlinear=add_nonlinear,
                add_unstructured=add_unstructured
            )

    end_time = datetime.now()
    logging.info(f"{compute_type.capitalize()} dataset generation completed in {end_time - start_time}.")

# ---------------------------
# Scenario Setup (Example)
# ---------------------------
if __name__ == "__main__":
    # Example usage:
    n_list = [100, 500, 1000]
    distribution_list = ["poisson", "gaussian_homo", "gaussian_hetero"]
    SNR_list = [1, 8]
    n_rep = 100  # for testing; can be 100 in real usage
    grid_size = 28
    n_core = mp.cpu_count()
    print(f"Number of cores: {n_core}")

    # Define linear coefficients alpha_l
    alpha_l = {
        0: [1, -1],      # For location: alpha_0 for X1, X2
        1: [-1, 1]     # For scale (or second dimension)
    }

    # Define some example nonlinear functions
    def nonlinear_effect_1_z1(z1):
        return scale_to_range(3 * np.sin(6 * z1))

    def nonlinear_effect_1_z2(z2):
        return scale_to_range(np.exp(5 * z2))

    def nonlinear_effect_2_z1(z1):
        return scale_to_range(np.cos(8 * z1))

    def nonlinear_effect_2_z2(z2):
        return scale_to_range(0.1 * np.sqrt(z2))

    # Put them into a dictionary: beta_nl[k][j]
    beta_nl = {
        0: [nonlinear_effect_1_z1, nonlinear_effect_1_z2],
        1: [nonlinear_effect_2_z1, nonlinear_effect_2_z2]
    }

    # Now call the main generator with add_nonlinear=False and add_unstructured=False
    scenarios_generate(
        n_list=n_list,
        distribution_list=distribution_list,
        SNR_list=SNR_list,
        grid_size=grid_size,
        alpha_l=alpha_l,
        beta_nl=beta_nl,
        n_rep=n_rep,
        n_cores=n_core,
        compute_type='parallel',   # or 'serial'
        add_linear=True,
        add_nonlinear=True,       # no nonlinear
        add_unstructured=False     # no unstructured
    )
