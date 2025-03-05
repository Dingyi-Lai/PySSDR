import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF
import tensorflow as tf
from PIL import Image
from tensorflow import keras

from keras import Input, Model
from keras.models import load_model, Sequential
from keras.layers import Dense, Flatten, ReLU

import torch
import torch.nn as nn
import torch.optim as optim

from time import time
from scipy.stats import poisson, gamma, norm
from scipy.optimize import minimize_scalar
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

# from sddr import Sddr  # Assuming you have pyssdr installed and configured correctly
import logging
from datetime import datetime
from itertools import product

# import torch
logging.basicConfig(level=logging.INFO)
import os
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# import the sddr module
from sddr import Sddr



def scale_to_range(data, lower=-1, upper=1):
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * (upper - lower) + lower

def save_with_var_name(var, var_name, var_type, save_path, scenario_index):
    if var_type == 'npy':
        np.save(f"{save_path}/{var_name}_{scenario_index}.npy", var)
    if var_type == 'keras':
        var.save(f"{save_path}/{var_name}_{scenario_index}.keras")
    if var_type == 'jpgs':
        images_path = f"{save_path}/{var_name}_{scenario_index}"
        os.makedirs(images_path, exist_ok=True)
        for idx, img in enumerate(var):
            # Normalize and convert to uint8
            normalized_img = (img * 255 / np.max(img)).astype(np.uint8)
            # Convert the NumPy array to an image
            normalized_img = Image.fromarray(normalized_img).convert("L")
            normalized_img.save(f"{images_path}/{var_name}_{scenario_index}_{idx}.jpg")
    if var_type == 'pth':
        var.save(f"{var_name}_{scenario_index}.pth")
    if var_type == 'df':
        var.to_csv(f"{save_path}/{var_name}_{scenario_index}.csv", index=False)

    logging.info(f"Saved {var_name} to {save_path}/{var_name}_{scenario_index}.npy")
    
def read_with_var_name(var_name, var_type, save_path, scenario_index):
    if var_type == 'npy':
        return np.load(f"{save_path}/{var_name}_{scenario_index}.npy")
    if var_type == 'keras':
        return tf.keras.models.load_model(f"{save_path}/{var_name}_{scenario_index}.keras")

# def plot_true_and_ci(true_effect, partial_effect, param, spline_index):
#     """
#     Plots the estimated partial effect with its 95% CI and overlays the true nonlinear effect.
    
#     Parameters:
#       - true_effect: 1D array of shape (n_samples,) for the true effect for this spline.
#       - partial_effect: tuple (feature, pred, ci950, ci951, ci250, ci251) from ssdr.eval.
#       - param: parameter name (e.g., 'loc' or 'scale' or 'rate').
#       - spline_index: index of the current spline.
#     """
#     feature, pred, ci950, ci951, _, _ = partial_effect

#     # Sort by feature for better plotting.
#     sort_idx = np.argsort(feature)
#     feature_sorted = np.array(feature)[sort_idx]
#     pred_sorted = np.array(pred)[sort_idx]
#     ci950_sorted = np.array(ci950)[sort_idx]
#     ci951_sorted = np.array(ci951)[sort_idx]
#     true_effect_sorted = np.array(true_effect)[sort_idx]

#     plt.figure(figsize=(8, 6))
#     plt.plot(feature_sorted, pred_sorted, label="Estimated Partial Effect", color="blue")
#     plt.fill_between(feature_sorted, ci950_sorted, ci951_sorted, color="blue", alpha=0.3, label="95% CI")
#     plt.scatter(feature_sorted, true_effect_sorted, color="red", marker="x", label="True Nonlinear Effect")
#     plt.title(f"Parameter: {param} - Spline {spline_index}")
#     plt.xlabel("Feature")
#     plt.ylabel("Effect")
#     plt.legend()
#     plt.show()
    
# ---------------------------
# Train Task Function for Parallel Execution
# ---------------------------
def predict_effects(scenario_index_folder, read_path, save_path, case, data, train_parameters,
                    num_knots, grid_size, output_dimension_dnn, 
                    add_linear=True, add_nonlinear=True,
                    add_unstructured=True, modify=True, ortho_manual=False):
    distribution, snr, method = case
    # os.makedirs(save_path, exist_ok=True)
    scenario_index = scenario_index_folder + f"_dist_{distribution}_SNR_{snr}"

    # a = read_with_var_name('a', 'npy', save_path, scenario_index)
    # etas = read_with_var_name('etas', 'npy', save_path, scenario_index)
    y = read_with_var_name('y', 'npy', read_path, scenario_index)
    data['Y'] = y
    # print(scenario_index)
    # print(data)
    # Train base SSDR model
    deep_models_dict = {
        'dnn': {
            'model': 
                # nn.Sequential(
                # nn.Flatten(1, -1),
                # nn.Linear(grid_size*grid_size,output_dimension_dnn),
                # nn.ReLU()
                
                nn.Sequential(
                nn.Flatten(1, -1),
                nn.Linear(grid_size*grid_size, 32, bias=False),
                nn.ReLU(),
                nn.Linear(32, 16),
                # nn.ReLU(),
                # nn.Linear(16, 1)
                ),
            
            'output_shape': output_dimension_dnn},
    }

    # provide the location and datatype of the unstructured data
    unstructured_data = {
    'Image' : {
        'path' : f"{read_path}/images_jpg_{scenario_index_folder}/",
        'datatype' : 'image'
        }
    }
    
    item_formula = ""
    if add_linear:
        item_formula += f" + X1 + X2"
    if add_nonlinear:
        item_formula += f" + spline(Z1, bs='bs', df={num_knots+3}) + spline(Z2, bs='bs', df={num_knots+3})"
    if add_unstructured:
        item_formula += f" + dnn(Image)"
        
        
    if distribution == "poisson":
        distribution_SSDR = "Poisson" # compatible form
        formulas = {'rate': f"~ 1"+item_formula}
        degrees_of_freedom = {'rate':num_knots+3}
    elif distribution == "gamma":
        distribution_SSDR = "Gamma" # compatible form
        formulas = {
        'loc': f"~ 1"+item_formula,
        'scale': '~ 1'
        }
        degrees_of_freedom = {'loc':num_knots+3, 'scale':num_knots+3}
    elif distribution == "gaussian_homo":
        distribution_SSDR = "Normal" # compatible form
        formulas = {
        'loc': f"~ 1"+item_formula, #
        'scale': '~ 1'
        }
        degrees_of_freedom = {'loc':num_knots+3, 'scale':num_knots+3}
    elif distribution == "gaussian_hetero":
        distribution_SSDR = "Normal" # compatible form
        formulas = {
        'loc': f"~ 1"+item_formula,
        'scale': f"~ 1"+item_formula,
        }
        degrees_of_freedom = {'loc':num_knots+3, 'scale':num_knots+3}

    train_parameters['degrees_of_freedom'] = degrees_of_freedom
    
    logging.info(f"Ready for the Sddr...")

    # model.fit([X_struct, X_unstruct], y, epochs=50, batch_size=32, verbose=1)
    if method == "point_estimates":
        # define your training hyperparameters
        ssdr = Sddr(output_dir=save_path,
            distribution=distribution_SSDR,
            formulas=formulas,
            deep_models_dict=deep_models_dict,
            train_parameters=train_parameters,
            modify=modify,
            ortho_manual = ortho_manual,
            use_spline_for_struct = False,
            n_knots = num_knots
            )
        # print(train_parameters['epochs'])
        scenario_index += f"_{method}"
        # model_path = f"{save_path}/ssdr_{scenario_index}.pth"
        # print(model_path)
        # if os.path.exists(model_path):
        #     # Here, final_epochs is your intended final epoch count (e.g., 300) 
        #     ssdr.load(model_path, data)
        #     ssdr.train(target="Y", structured_data=data, resume=True)
        # else:  
        ssdr.train(structured_data=data,
            target="Y",
            # unstructured_data = unstructured_data,
            plot=False)
        save_with_var_name(ssdr, 'ssdr', 'pth', save_path, scenario_index)
        logging.info(f"Save the model {scenario_index} in {save_path}")
        
        # Create an empty list to store the result rows.
        results = []

        # Loop over each parameter group in degrees_of_freedom.
        for k in degrees_of_freedom.keys():
            # Get the coefficient dictionary for parameter group k.
            # This dictionary is expected to have keys corresponding to feature names.
            
            # Combine the features (if you want to process both kinds together)
            coeff_dict = ssdr.coeff(k)
            
            for feature in coeff_dict.keys():
                # Extract the point estimate for the feature.
                # (Assuming ssdr.coeff(k)[feature] returns a list/array where the first element is the estimate.)                
                # Append a dictionary with the desired columns.
                results.append({
                    'scenario_index': scenario_index,  # scenario_index should be defined in your code
                    'param_y': k,
                    'param_eta': feature,
                    'value': coeff_dict[feature]
                })

        # Convert the list of dictionaries to a DataFrame.
        df_results = pd.DataFrame(results)
        save_with_var_name(df_results, 'point_estimates', 'df', save_path, scenario_index)
        logging.info(f"Save the estimates for {scenario_index} in {save_path}")
        return df_results
    # if method == "point_estimates":
    #     # define your training hyperparameters and train the model
    #     ssdr = Sddr(output_dir=save_path,
    #                 distribution=distribution_SSDR,
    #                 formulas=formulas,
    #                 deep_models_dict=deep_models_dict,
    #                 train_parameters=train_parameters,
    #                 modify=True)
    #     scenario_index += f"_{method}"
    #     model_path = f"{save_path}/ssdr_{scenario_index}.pth"
    #     print(model_path)
    #     if os.path.exists(model_path):
    #         # load and resume training
    #         ssdr.load(model_path, data)
    #         ssdr.train(target="Y", structured_data=data, resume=True)
    #     else:
    #         ssdr.train(structured_data=data,
    #                 target="Y",
    #                 unstructured_data=unstructured_data,
    #                 plot=True)
    #         save_with_var_name(ssdr, 'ssdr', 'pth', save_path, scenario_index)
        
    #     # Create an empty list to store result rows.
    #     results = []
        
    #     # Loop over each parameter group in degrees_of_freedom.
    #     for k in degrees_of_freedom.keys():
    #         # Get the coefficient dictionary (point estimates) for the structured head.
    #         coeff_dict = ssdr.coeff(k)
            
    #         # Also get the corresponding structured weights and latent features from the deep branch.
    #         # (get_weights_and_latent_features returns a dict with keys 'structured_weights' and 'latent_features'.)
    #         extra_info = ssdr.get_weights_and_latent_features(k, data)
            
    #         # Loop over each term in the coefficient dictionary.
    #         for term in coeff_dict.keys():
    #             results.append({
    #                 'scenario_index': scenario_index,  # the current scenario index
    #                 'param_y': k,
    #                 'param_eta': term,
    #                 'value': coeff_dict[term],
    #                 'structured_weight': extra_info['structured_weights'][term],
    #                 # If latent_features is a tensor, convert to list for storage.
    #                 'latent_features': (extra_info['latent_features'].cpu().numpy().tolist() 
    #                                     if extra_info['latent_features'] is not None 
    #                                     else None)
    #             })
        
    #     # Convert the list of dictionaries to a DataFrame.
    #     df_results = pd.DataFrame(results)
    #     save_with_var_name(df_results, 'point_estimates', 'df', save_path, scenario_index)
    #     return df_results

    # Deep ensemble
    # elif method == "deep_ensemble":
    #     # Train ensemble of models
    #     n_ensemble = 5
        # ensemble_models = []
        # for i in range(n_ensemble):
        #     # define your training hyperparameters
        #     train_parameters = {
        #         'batch_size': 3,
        #         'epochs': 100,
        #         'degrees_of_freedom': degrees_of_freedom,
        #         'optimizer' : optim.Adam,
        #         'val_split': 0.15,
        #         'early_stop_epsilon': 0.001,
        #     }
        #     ensemble_model = build_ssdr_model(X_struct.shape[1], X_unstruct.shape[1])
        #     ensemble_model.fit([X_struct, X_unstruct], y, epochs=50, verbose=0)
        #     ensemble_models.append(ensemble_model)
        
        # # Predict and compute ensemble uncertainty
        # ensemble_predictions = np.array([
        #     model.predict([X_struct_eval, X_unstruct_eval]) for model in ensemble_models
        # ])
        # mean_predictions = np.mean(ensemble_predictions, axis=0)
        # uncertainty = np.std(ensemble_predictions, axis=0)
    
    # Dropout sampling
    # elif method == "dropout_sampling":
    #     # define your training hyperparameters
    #     ssdr = Sddr(output_dir=save_path,
    #         distribution=distribution_SSDR,
    #         formulas=formulas,
    #         deep_models_dict=deep_models_dict,
    #         train_parameters=train_parameters,
    #         modify=True
    #         )
    #     # print(train_parameters['epochs'])
    #     scenario_index += f"_{method}"
    #     model_path = f"{save_path}/ssdr_{scenario_index}.pth"
    #     if os.path.exists(model_path):
    #         # Here, final_epochs is your intended final epoch count (e.g., 300) 
    #         ssdr.load(model_path, data)
    #         ssdr.train(target="Y", structured_data=data, resume=True)
    #     else:  
    #         ssdr.train(structured_data=data,
    #             target="Y",
    #             unstructured_data = unstructured_data,
    #             plot=True)
            
    #         save_with_var_name(ssdr, 'ssdr', 'pth', save_path, scenario_index)

        
    #     eval_dict = {}
    #     for k in degrees_of_freedom.keys():
    #         eval_results = ssdr.eval(k, plot=False)
    #         eval_dict[k] = eval_results
        
    #     param_to_index = {"rate": 0, "loc": 0, "scale": 1}
    
    #     # Compute coverage rates for each parameter.
    #     coverage_rates = {}
    #     for param, partial_effects in eval_dict.items():
    #         if len(partial_effects)>0:
    #             coverage_rates[param] = {}
    #             # For Gaussian (or gamma) cases, we assume true_nonlinear_effects is a dict with keys matching the parameter names.
    #             # For Poisson, true_nonlinear_effects is a list.
    #             true_effects = true_nonlinear_effects[:,:,param_to_index[param]]

    #             # for idx, effect in enumerate(partial_effects):
    #             #     if len(effect) == 6:
    #             #         _, _, ci950, ci951, _, _ = effect
    #             #         true_effect = true_effects[idx,:]
    #             #         covered = np.logical_and(true_effect >= ci950, true_effect <= ci951)
    #             #         coverage_rate = np.mean(covered)
    #             #         coverage_rates[param][f'spline_{idx}'] = coverage_rate
    #             #     else:
    #             #         coverage_rates[param][f'spline_{idx}'] = None
    #             for idx, effect in enumerate(partial_effects):
    #                 if len(effect) == 6:
    #                     feature, _, ci950, ci951, _, _ = effect
                        
    #                     # Get the corresponding true effect for this spline; shape: (n_samples, )
    #                     true_effect = true_effects[idx, :]
    #                     plot_true_and_ci(true_effect, effect, param, idx)
    #                     # Sort both the feature and true effect to ensure proper alignment.
    #                     sort_idx = np.argsort(feature)
    #                     sorted_feature = np.array(feature)[sort_idx]
    #                     sorted_ci950 = ci950[sort_idx]
    #                     sorted_ci951 = ci951[sort_idx]
    #                     sorted_true_effect = true_effect[sort_idx]
                        
    #                     # Now compute coverage.
    #                     covered = np.logical_and(sorted_true_effect >= sorted_ci950, sorted_true_effect <= sorted_ci951)
    #                     coverage_rate = np.mean(covered)
    #                     coverage_rates[param][f'spline_{idx}'] = coverage_rate
    #                 else:
    #                     coverage_rates[param][f'spline_{idx}'] = None

        
    #     print("Coverage rates for replicate:")
    #     for param, cov_dict in coverage_rates.items():
    #         print(f"Parameter {param}:")
    #         for key, val in cov_dict.items():
    #             if val is not None:
    #                 print(f"  {key}: {val:.2%}")
    #             else:
    #                 print(f"  {key}: N/A")
        
    #     return coverage_rates
    
    # # Last-layer inference
    # elif method == "last_layer":
    #     mean_predictions, uncertainty = last_layer_inference(
    #         model, X_struct_eval, X_unstruct_eval
    #     )
    
    else:
        raise ValueError("Unsupported method. Choose from 'point_estimates', 'deep_ensemble', 'dropout_sampling', or 'last_layer'.")
    
    
    
    
    
    # # Calculate confidence intervals and coverage rates
    # confidence_level = 0.95
    # z = norm.ppf(1 - (1 - confidence_level) / 2)
    # lower_bound = mean_predictions - z * uncertainty
    # upper_bound = mean_predictions + z * uncertainty
    # coverage_rate = np.mean((y_eval >= lower_bound) & (y_eval <= upper_bound))
    
    # return {
    #     "mean_predictions": mean_predictions,
    #     "uncertainty": uncertainty,
    #     "coverage_rate": coverage_rate,
    #     "lower_bound": lower_bound,
    #     "upper_bound": upper_bound
    # }

    # train_per_case(scenario, save_path, distribution, snr, X, images, y, epochs=50, batch_size=32, learning_rate=0.01, verbose=0)

    # return sddr
    
def training_task(n_sample, distribution_list, SNR_list, method_list, grid_size, train_parameters, 
                  num_knots, n_rep, read_path, save_path, compute_type='parallel',
                  add_linear=True, add_nonlinear=True, add_unstructured=True, modify=True, ortho_manual=False):
    
    # Set random seed for reproducibility
    np.random.seed(n_sample+n_rep)
    logging.info(f"Reading dataset: Number of obs {n_sample} | Replication {n_rep}")

    scenario_index = f"n_{n_sample}_rep_{n_rep}"
    parts = []
    column_names = []

    if add_linear:
        X = read_with_var_name('X', 'npy', read_path, scenario_index)
        X_transposed = X.T  # e.g. shape (10, 2)
        parts.append(X_transposed)
        column_names.extend(['X1', 'X2'])

    if add_nonlinear:
        Z = read_with_var_name('Z', 'npy', read_path, scenario_index)
        # nonlinear_effects = read_with_var_name('nonlinear_effects', 'npy', read_path, scenario_index)
        Z_transposed = Z.T  # e.g. shape (10, 2)
        parts.append(Z_transposed)
        column_names.extend(['Z1', 'Z2'])

    # If both parts are false, you can decide what to do (e.g., throw an error).
    if not parts:
        raise ValueError("Neither linear nor nonlinear data was selected!")

    combined_data = np.hstack(parts)
    df = pd.DataFrame(combined_data, columns=column_names)

    U_k = None
    output_dimension_dnn = 0
    
    if add_unstructured:
        df['Image'] = [f'images_jpg_{scenario_index}_{i}.jpg' for i in range(len(df))]
        # print(df)
        # images = read_with_var_name('images', 'npy', save_path, scenario_index)
        
        # dnn_model = read_with_var_name('dnn_model', 'keras', save_path, scenario_index)
        # unstructured_effects = read_with_var_name('unstructured_effects', 'npy', save_path, scenario_index)
        U_k = read_with_var_name('U_k', 'npy', read_path, scenario_index)
        # psi_k = read_with_var_name('psi_k', 'npy', save_path, scenario_index)
        # b_k = read_with_var_name('b_k', 'npy', save_path, scenario_index)
        output_dimension_dnn = U_k.shape[1]
    
    combinations = list(product(distribution_list, SNR_list, method_list))
    logging.info(f"Ready for the ThreadPoolExecutor...")
    # For each combination, call predict_effects.
    # coverage_results = {}
    if compute_type == 'parallel':
        def process_combination(c):
            df_results = predict_effects(scenario_index, read_path, save_path, c, df, train_parameters,
                                             num_knots, grid_size, output_dimension_dnn,
                                             add_linear, add_nonlinear, add_unstructured, modify, ortho_manual)
            return df_results

        with ThreadPoolExecutor() as executor:
            _ = executor.map(process_combination, combinations)
        
    if compute_type == 'serial':
        for c in combinations:
            predict_effects(scenario_index, read_path, save_path, c, df, train_parameters,
                                             num_knots, grid_size, output_dimension_dnn,
                                             add_linear, add_nonlinear, add_unstructured, modify, ortho_manual)    

# ---------------------------
# Parallel Execution Function
# ---------------------------

def uq_comparison(n_list, distribution_list, SNR_list, method_list, grid_size,
                  train_parameters_list, num_knots, n_rep, n_cores, save_path,
                  compute_type='parallel', add_linear=True, add_nonlinear=True, add_unstructured=True, modify=True, ortho_manual=False):
    logging.info(f"Starting {compute_type} training...")
    start_time = datetime.now()
    replicates = [(i, r) for i in n_list for r in range(n_rep)]
    read_path = "../data_generation/output_nonlinear"
    # read_path = os.environ.get("READ_PATH", "../data_generation/output")

    os.makedirs(save_path, exist_ok=True)
    
    # results = []
    if compute_type == 'parallel':
        with mp.Pool(n_cores) as pool:
            pool.starmap(
                training_task,
                [(i, distribution_list, SNR_list, method_list, grid_size, train_parameters_list[n_list.index(i)], 
                  num_knots, r, read_path, save_path, compute_type, add_linear, add_nonlinear, add_unstructured, modify, ortho_manual)
                 for (i, r) in replicates]
            )
    elif compute_type == 'serial':
        for (i, r) in replicates:
            training_task(n_sample=i, distribution_list=distribution_list, 
                                         SNR_list=SNR_list, method_list=method_list, grid_size=grid_size, 
                                         train_parameters=train_parameters_list[n_list.index(i)], 
                                         num_knots=num_knots, n_rep=r, read_path=read_path, save_path=save_path,
                                         compute_type=compute_type, add_linear=add_linear, add_nonlinear=add_nonlinear,
                                         add_unstructured=add_unstructured, modify=modify, ortho_manual=ortho_manual)
    end_time = datetime.now()
    logging.info(f"Training completed in {end_time - start_time}.")
    # return results

# ---------------------------
# Scenario Setup
# ---------------------------
if __name__ == '__main__':

    # n_list = [100, 500, 1000]
    n_list = [1000]
    distribution_list = ["gaussian_homo"] # "gamma", "poisson", 
    SNR_list=[1, 8] #,8]
    # method_list = ['deep_ensemble', 'dropout_sampling', 'last_layer']
    method_list = ['point_estimates']
    n_rep=100
    grid_size = 28
    n_core = mp.cpu_count()
    print(f"Number of cores: {n_core}")
    # define output directory
    nbatch = 32

    # # For n=100:
    # train_parameters_small = {
    #     'batch_size': 16,              # Smaller batch size due to limited sample size.
    #     'epochs': 100,
    #     # 'degrees_of_freedom': {'rate': 3},  # For Poisson; adjust accordingly for other distributions.
    #     'optimizer': optim.Adam,
    #     'val_split': 0.20,             # Possibly a higher split for very small samples.
    #     'early_stop_epsilon': 0.001,
    #     # 'dropout_rate': 0.01           # Start with 0.01; consider 0.01-0.05 range.
    # }

    # # For n=500 and n=1000:
    train_parameters_large = {
        'batch_size': nbatch,              # Larger batch size as more data is available.
        'epochs': 100,
        # 'degrees_of_freedom': {'rate': 3},  # Or {'loc': 3, 'scale': 3} for Gaussian cases.
        'optimizer': optim.Adam,
        'val_split': 0.15,
        'early_stop_epsilon': 0.001,
        # 'dropout_rate': 0.01           # Can experiment with 0.01 to 0.05.
    }
    # For n=100:
    # train_parameters_100 = {
    #     'batch_size': 100,              # Smaller batch size due to limited sample size.
    #     'epochs': 100,
    #     # 'degrees_of_freedom': {'rate': 3},  # For Poisson; adjust accordingly for other distributions.
    #     'optimizer': optim.Adam,
    #     'val_split': 0.20,             # Possibly a higher split for very small samples.
    #     'early_stop_epsilon': 0.001,
    #     # 'dropout_rate': 0.01           # Start with 0.01; consider 0.01-0.05 range.
    # }

    # # For n=500
    # train_parameters_500 = {
    #     'batch_size': 500,              # Larger batch size as more data is available.
    #     'epochs': 100,
    #     # 'degrees_of_freedom': {'rate': 3},  # Or {'loc': 3, 'scale': 3} for Gaussian cases.
    #     'optimizer': optim.Adam,
    #     'val_split': 0.15,
    #     'early_stop_epsilon': 0.001,
    #     # 'dropout_rate': 0.01           # Can experiment with 0.01 to 0.05.
    # }
    
    # # For n=1000:
    # train_parameters_1000 = {
    #     'batch_size': 1000,              # Larger batch size as more data is available.
    #     'epochs': 100,
    #     # 'degrees_of_freedom': {'rate': 3},  # Or {'loc': 3, 'scale': 3} for Gaussian cases.
    #     'optimizer': optim.Adam,
    #     'val_split': 0.15,
    #     'early_stop_epsilon': 0.001,
    #     # 'dropout_rate': 0.01           # Can experiment with 0.01 to 0.05.
    # }

    num_knots = 6
    # train_parameters_list = [train_parameters_small, train_parameters_large, train_parameters_large]
    train_parameters_list = [train_parameters_large] # , train_parameters_500, train_parameters_1000

    # save_path = './outputs_w_unstructured'
    # save_path = os.path.join(os.environ["TMPDIR"], "outputs_w_unstructured")
    # uq_comparison(n_list, distribution_list, SNR_list, method_list, grid_size, train_parameters_list,
    #                         num_knots,
    #                         n_rep, n_cores=n_core, save_path=save_path, compute_type='parallel',
    #                         add_unstructured=True, modify=True, ortho_manual=False) # parallel
    
    save_path = './outputs_structured_nknots_6_batch_32'
    # save_path = os.path.join(os.environ["TMPDIR"], "outputs_nonlinear_nknots_"+str(num_knots)+"_batch_"+str(nbatch))
    uq_comparison(n_list, distribution_list, SNR_list, method_list, grid_size, train_parameters_list,
                            num_knots,
                            n_rep, n_cores=n_core, save_path=save_path, compute_type='parallel',
                            add_linear=True, add_nonlinear=True, add_unstructured=False, modify=False, ortho_manual=False) # parallel
    
    # save_path = './outputs_with_unstructured'
    # uq_comparison(n_list, distribution_list, SNR_list, method_list, grid_size, train_parameters_list,
    #                         num_knots,
    #                         n_rep, n_cores=n_core, save_path=save_path, compute_type='parallel',
    #                         add_unstructured=True, modify=True, ortho_manual=False) # parallel
    
    # uq_comparison(n_list, distribution_list, SNR_list, method_list, grid_size, train_parameters_list,
    #                         num_knots,
    #                         n_rep, n_cores=n_core, save_path=save_path, compute_type='serial',
    #                         add_unstructured=False, modify=True, ortho_manual=False) # parallel
    
    # # Aggregate the coverage results across replicates.
    # aggregated_coverage = {}
    # for n_sample, rep, cov_dict in results:
    #     for key, cov in cov_dict.items():
    #         if key not in aggregated_coverage:
    #             aggregated_coverage[key] = []
    #         aggregated_coverage[key].append(cov)

    # # Compute the average coverage rate for each scenario and parameter.
    # for key, cov_list in aggregated_coverage.items():
    #     avg_cov = {}
    #     for param in cov_list[0]:
    #         replicate_means = []
    #         for replicate_cov in cov_list:
    #             # replicate_cov[param] is a dictionary of spline coverage rates,
    #             # e.g. {'spline_0': 0.85, 'spline_1': 0.90}
    #             spline_rates = [r for r in replicate_cov.get(param, {}).values() if r is not None]
    #             if spline_rates:
    #                 replicate_means.append(np.mean(spline_rates))
    #         avg_cov[param] = np.mean(replicate_means) if replicate_means else None
    #     print(f"Scenario {key}: Average Coverage Rates: {avg_cov}")

    # # Optionally, save the aggregated coverage results.
    # save_with_var_name(aggregated_coverage, "aggregated_coverage", "npy", save_path, "all_scenarios")