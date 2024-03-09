##############################
###--------PACKAGES--------###
##############################

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
import GPy
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.kernel_ridge import KernelRidge

def read_cgns_coordinates(file_path):
    """Access the .cgns files
    
    """
    with h5py.File(file_path, 'r') as file:
        # We retrieve coordinate by coordinate.
        # ! Notice the space before the data. This is due to the naming in the files themselves.
        x = np.array(file['Base_2_3/Zone/GridCoordinates/CoordinateX'].get(' data'))
        y = np.array(file['Base_2_3/Zone/GridCoordinates/CoordinateY'].get(' data'))
        z = np.array(file['Base_2_3/Zone/GridCoordinates/CoordinateZ'].get(' data'))

    return x, y, z

def import_blades(split: list, sample_size:int, path:str = None, sample_fn = None) -> list:
    """Imports the blade which id are in split..

    Args:
        split(list): list of blade to consider
        path(str): add to the path
        sample_fn(func): function that subsample a blade

    Returns:
        list: list of np.arrays. One np.array represents a blade.
    """

    padded_split = [str(i).zfill(9) for i in split]

    distributions = []

    for id in padded_split:
        ## File paths Personal Computer
        cgns_file_path = f'Rotor37/dataset/samples/sample_{id}/meshes/mesh_000000000.cgns'
        if path is not None:
            cgns_file_path = path + cgns_file_path
        ## Computing the coordinates
        x, y, z = read_cgns_coordinates(cgns_file_path)
        blade = np.column_stack((x, y, z))
        ## Subsampling if necessary
        if sample_fn is not None:
            indices = sample_fn(blade, size = sample_size)
            blade = blade[indices]
        else:
            pass
        ## Adding to our data
        distributions.append(blade)
    
    return distributions

def import_sinkhornPotentials_and_scalars(problem, problem_txt, test, 
                             ref_measure_txt: str, ref_measure_size: int, mu, sigma, center, radius,
                             epsilon, epsilon_txt: str,
                             subsampling_size: int, subsampling_size_txt: str,
                             subsampling_method, subsampling_method_txt: str,
                             precomputed_indices,
                             path_to_rotor37, path_to_saved_data):
    """Imports the blade which id are in split..

    Args:
        split(list): list of blade to consider
        path(str): add to the path
        sample_fn(func): function that subsample a blade

    Returns:
        list: list of np.arrays. Sinkhorn Potentials.
        list: list of int. Efficiency
        list: list of int. Omega
        list: list of int. P
        list: list of int. Massflow
        list: list of int. Compression Ratio
        dict: Metadatas
    """
    # Import Sinkhorn Potentials
    sinkhorn_path = path_to_saved_data + problem_txt + "/" + subsampling_method_txt + "/" + subsampling_size_txt + "/" + "sinkhorn_potentials_" + problem_txt + "_" + subsampling_method_txt + "_" +  subsampling_size_txt + "_epsilon" + epsilon_txt + "_" + ref_measure_txt + str(ref_measure_size) + ".npy"
    sinkhorn_potentials = np.load(sinkhorn_path)

    metadata_path = path_to_saved_data + problem_txt + "/" + subsampling_method_txt + "/" + subsampling_size_txt + "/" + "sinkhorn_metadata_" + problem_txt + "_" + subsampling_method_txt + "_" +  subsampling_size_txt + "_epsilon" + epsilon_txt + "_" +  ref_measure_txt + str(ref_measure_size) + ".json"
    with open(metadata_path) as f:
        metadata = json.load(f)
        
    padded_split = [str(i).zfill(9) for i in problem]

    # Import scalars
    efficiency = []
    omega = []
    P = []
    compression_ratio = []
    massflow = [] 
    for id in padded_split:
        coefficient_file_path = f'Rotor37/dataset/samples/sample_{id}/scalars.csv'
        scalars = pd.read_csv(coefficient_file_path)
        ## Adding to our data
        efficiency.append(scalars["Efficiency"][0])
        omega.append(scalars["Omega"][0])
        P.append(scalars["P"][0])
        compression_ratio.append(scalars["Compression_ratio"][0])
        massflow.append(scalars["Massflow"][0])
    
    return sinkhorn_potentials, efficiency, omega, P, compression_ratio, massflow, metadata

def random_subsample_fn(X: np.ndarray, size: int) -> np.ndarray:
    """Selects lines in the input sample randomly.

    Args:
        X(np.ndarray): input sample
        size(int): number of lines to select

    Returns:
        np.ndarray: array of selected lines
    """
    indices = np.random.choice(len(X), size=size, replace=False)
    
    return indices

def mmd_subsample_fn(X: np.ndarray, size: int) -> np.ndarray:
    """Selects lines in the input sample by greedily minimizing the maximum mena discrepancy.

    Args:
        X(np.ndarray): input sample
        size(int): number of lines to select

    Returns:
        np.ndarray: array of selected lines
    """

    n = X.shape[0]
    assert size<=n

    idx = np.zeros(size, dtype=np.int64)

    k0 = np.zeros((n,size))
    k0[:,0] = 2.0*np.sqrt(np.sum(X.T*X.T, axis=0))

    dist_matrix = cdist(X,X)
    gram_matrix = np.linalg.norm(X, axis=1)[:,None] + np.linalg.norm(X, axis=1)[None] - dist_matrix
    k0_mean = np.mean(gram_matrix, axis=1)

    idx[0] = np.argmin(k0[:,0]-2.0*k0_mean)
    for i in range(1,size):
        x_ = X[idx[i-1]][None]
        dist = (X-x_).T
        k0[:,i] = -np.sqrt(np.sum(dist*dist, axis=0)) + np.sqrt(np.sum(x_.T*x_.T, axis=0)) + np.sqrt(np.sum(X.T*X.T, axis=0))
        idx[i] = np.argmin(k0[:,0] + 2.0*np.sum(k0[:,1:(i+1)], axis=1) - 2.0*(i+1)*k0_mean)
    return idx

def custom_grid_search(param_grid, x_train, y_train, sampling_size, train_size:float = 0.8):
    """Performs cross validation on the kernels parameters.

    Args:


    Returns:
        the model
        the score on validation sample
        the hyperparameters chosen
    """    
    best_score = -np.inf
    best_params = None

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size = train_size, random_state = 42)

    sinkhorn_train = x_train[:, 0:sampling_size]
    scalars_train = x_train[:, sampling_size:]

    sinkhorn_val = x_val[:, 0:sampling_size]
    scalars_val = x_val[:, sampling_size:]

    for params in param_grid:
        gamma = params['gamma']
        gamma1 = params['gamma1']
        gamma2 = params['gamma2']
        alpha = params['alpha']

        krr = KernelRidge(kernel="precomputed", alpha = alpha)

        kernel_sinkhorn = kernels.RBF(length_scale=np.array(gamma))
        kernel_scalars = kernels.RBF(length_scale=np.array([gamma1, gamma2]))
        
        kernel_matrix_sinkhorn_train = kernel_sinkhorn(sinkhorn_train)
        kernel_matrix_scalars_train = kernel_scalars(scalars_train)
        k_train = kernel_matrix_sinkhorn_train * kernel_matrix_scalars_train
        
        krr.fit(X=k_train, y=y_train)
        
        kernel_matrix_sinkhorn_val = kernel_sinkhorn(sinkhorn_val, sinkhorn_train)
        kernel_matrix_scalars_val = kernel_scalars(scalars_val, scalars_train)
        k_val = kernel_matrix_sinkhorn_val * kernel_matrix_scalars_val
        
        predictions = krr.predict(X=k_val)
        score = -mean_squared_error(y_true=y_val, y_pred=predictions)
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_model = krr
    
    return best_model, best_score, best_params


def preprocessing(sinkhorn_potentials, list_of_X, list_of_Y, train_size):
    """Preprocess of all variables of the problem
    """

    for i in range(len(list_of_X)):
        list_of_X[i] = np.array(list_of_X[i]).reshape(-1, 1)

    for i in range(len(list_of_Y)):
        list_of_Y[i] = np.array(list_of_Y[i]).reshape(-1, 1)
    
    output_scalars = np.hstack(list_of_Y)
    feature_matrix = np.hstack([sinkhorn_potentials] + list_of_X)

    x_train, x_test, y_train, y_test = train_test_split(feature_matrix, output_scalars, train_size = train_size, random_state = 40)

    nb_potentials = sinkhorn_potentials.shape[1]

    sinkhorn_train = x_train[: , 0:nb_potentials]
    scalars_train = x_train[: , nb_potentials:]
    normalize = StandardScaler().fit(scalars_train)
    scalars_train = normalize.transform(scalars_train)
    x_train = np.hstack((sinkhorn_train, scalars_train))

    sinkhorn_test = x_test[: , 0:nb_potentials]
    scalars_test = x_test[: , nb_potentials:]
    # Apply the normalizer on the test data
    scalars_test = normalize.transform(scalars_test)
    x_test = np.hstack((sinkhorn_test, scalars_test))

    return x_train, x_test, y_train, y_test

def plot_reg(true, pred):
    plt.figure()

    sns.set_theme(context='paper', font_scale=1.5)
    sns.scatterplot(x=true, y=pred, color='blue', alpha=0.5)

    ## Adding the x=y line and the text
    plt.plot([min(true), max(true)], [min(true), max(true)], linestyle='--', color='red', linewidth=2, alpha = 0.8)

    plt.title('Predicted vs. Actual Values of Y')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')

    plt.show()