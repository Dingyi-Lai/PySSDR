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