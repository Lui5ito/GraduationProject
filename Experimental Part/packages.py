# General purpose packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Specifically for the Rotor37 Dataset importation and visualization
import h5py
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D

# For saving metadata
import json

# For optimized subsampling
from scipy.spatial.distance import cdist

# General purpose machine learning
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, train_test_split
from sklearn.gaussian_process import kernels
from sklearn.preprocessing import StandardScaler

# Machine learning models
import sklearn.kernel_ridge
import GPy
import catboost
import xgboost as xgb
from sklearn.kernel_ridge import KernelRidge