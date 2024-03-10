from packages import *

##############################
###--------SINKHORN--------###
##############################

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


def trainingData_preprocessing(sinkhorn_potentials, list_of_X, list_of_Y):
    """Preprocess of all variables of the problem
    """

    for i in range(len(list_of_X)):
        list_of_X[i] = np.array(list_of_X[i]).reshape(-1, 1)

    for i in range(len(list_of_Y)):
        list_of_Y[i] = np.array(list_of_Y[i]).reshape(-1, 1)
    
    y_train = np.hstack(list_of_Y)
    x_train = np.hstack([sinkhorn_potentials] + list_of_X)

    sinkhorn_length = sinkhorn_potentials.shape[1]

    sinkhorn_train = x_train[: , 0:sinkhorn_length]
    scalars_train = x_train[: , sinkhorn_length:]
    normalize = StandardScaler().fit(scalars_train)
    scalars_train = normalize.transform(scalars_train)
    x_train = np.hstack((sinkhorn_train, scalars_train))

    return x_train, y_train, normalize

def inferenceData_preprocessing(sinkhorn_potentials, list_of_X, list_of_Y, train_normalizer):
    """Preprocess of all variables of the problem
    """

    for i in range(len(list_of_X)):
        list_of_X[i] = np.array(list_of_X[i]).reshape(-1, 1)

    for i in range(len(list_of_Y)):
        list_of_Y[i] = np.array(list_of_Y[i]).reshape(-1, 1)
    
    y_test = np.hstack(list_of_Y)
    x_test = np.hstack([sinkhorn_potentials] + list_of_X)

    sinkhorn_length = sinkhorn_potentials.shape[1]

    sinkhorn_test = x_test[: , 0:sinkhorn_length]
    scalars_test = x_test[: , sinkhorn_length:]
    scalars_test = train_normalizer.transform(scalars_test)
    x_test = np.hstack((sinkhorn_test, scalars_test))

    return x_test, y_test


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

##############################
###--IMPORTING SAVED DATA--###
##############################
    
def import_train_data(problem, problem_txt, test, 
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


def import_test_data(problem, problem_txt, test, 
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
    sinkhorn_path = path_to_saved_data + problem_txt + "/" + subsampling_method_txt + "/" + subsampling_size_txt + "/" + "sinkhorn_potentials_test_" + problem_txt + "_" + subsampling_method_txt + "_" +  subsampling_size_txt + "_epsilon" + epsilon_txt + "_" + ref_measure_txt + str(ref_measure_size) + ".npy"
    sinkhorn_potentials = np.load(sinkhorn_path)

    metadata_path = path_to_saved_data + problem_txt + "/" + subsampling_method_txt + "/" + subsampling_size_txt + "/" + "sinkhorn_metadata_test_" + problem_txt + "_" + subsampling_method_txt + "_" +  subsampling_size_txt + "_epsilon" + epsilon_txt + "_" +  ref_measure_txt + str(ref_measure_size) + ".json"
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
        #efficiency.append(scalars["Efficiency"][0])
        omega.append(scalars["Omega"][0])
        P.append(scalars["P"][0])
        #compression_ratio.append(scalars["Compression_ratio"][0])
        #massflow.append(scalars["Massflow"][0])
    
    return sinkhorn_potentials, efficiency, omega, P, compression_ratio, massflow, metadata



##############################
###--------MODELLING-------###
##############################

#################
###--GENERAL--###
#################

def train(name: str, x_train, y_train, parameters_grid, sinkhorn_length):
    if name == "krr":
        krr_model = trainKRR(x_train, y_train, parameters_grid, sinkhorn_length)
        return krr_model
    if name == "gp":
        gp_model = trainGP(x_train, y_train, sinkhorn_length)
        return gp_model
    if name == "catboost":
        catboost_model = trainCatBoost(x_train, y_train, parameters_grid, sinkhorn_length)
        return catboost_model
    else:
        raise ValueError(f"The model {name} asked has not been coded yet.")
    
def inference(name, model, x_test):
    if name == "krr":
        krr_predictions = inferenceKRR(model, x_test)
        return krr_predictions
    if name == "gp":
        gp_predictions = inferenceGP(model, x_test)
        return gp_predictions
    if name == "catboost":
        catboost_predictions = inferenceCatBoost(model, x_test)
        return catboost_predictions
    else:
        raise ValueError(f"The model {name} asked has not been coded yet.")

#################
###----KRR----###
#################

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

def trainKRR(x_train, y_train, parameters_grid, sinkhorn_length):
    """Train a Kernel Ridge Regression model. This performs cross validation as well.
    This can be a moultioutputs model as well as a singleoutput model.

    A good addition would be the choice of kernels in function's parameters.
    """

    best_model, _, best_params = custom_grid_search(ParameterGrid(parameters_grid), x_train, y_train, sampling_size = sinkhorn_length, train_size = 0.8)

    gamma = best_params['gamma']
    gamma1 = best_params['gamma1']
    gamma2 = best_params['gamma2']

    kernel_sinkhorn = kernels.RBF(length_scale=np.array(gamma))
    kernel_scalars = kernels.RBF(length_scale=np.array([gamma1, gamma2]))
    
    ## Train Kernel Matrix
    sinkhorn_train = x_train[: , 0:sinkhorn_length]
    scalars_train = x_train[: , sinkhorn_length:]
    kernel_matrix_sinkhorn_train = kernel_sinkhorn(sinkhorn_train)
    kernel_matrix_scalars_train = kernel_scalars(scalars_train)
    k_train = kernel_matrix_sinkhorn_train*kernel_matrix_scalars_train

    ## Train the Kernel Ridge Regression the "best_model" objects contains the alpha parameter.
    best_model.fit(X = k_train, y = y_train)

    return best_model

def inferenceKRR(model, x_test, x_train, normalize):
    """For a given krr, predicts.
    """
    if type(model) != sklearn.kernel_ridge.KernelRidge:
        raise TypeError("The model given is not a kernel ridge regression from sklearn.")
    predictions = model.predict(x_test)
    return predictions

#################
###-----GP----###
#################

def trainGP(x_train, y_train, sinkhorn_length):
    """Train a Gaussian Process model. This performs cross validation as well.
    This can be a multioutputs GP and a singleoutput GP.

    A good addition would be to put the kernels in the function's argument.
    """
    kernel_sinkhorn = GPy.kern.RBF(input_dim = sinkhorn_length, active_dims = list(range(0, sinkhorn_length)), ARD = True)
    kernel_scalars = GPy.kern.RBF(input_dim = 2, active_dims = list(range(sinkhorn_length, x_train.shape[1])), ARD = True)
    kernel_product = kernel_sinkhorn.prod(kernel_scalars, name='productKernel')

    model = GPy.models.GPRegression(x_train, y_train, kernel_product, normalizer=False, noise_var=1.0)
    model.optimize_restarts(num_restarts = 6, messages = False, max_iters = 1000)

    return model

def inferenceGP(model, x_test):
    """For a given trained gaussian process model, we infere the testing data. 
    """

    if type(model) != GPy.models.gp_regression.GPRegression:
        raise TypeError("The model given is not a gaussian process from GPy package.")
    
    predictions, _ = model.predict(x_test)
    return predictions

#################
###--CATBOOST--##
#################

def trainCatBoost(x_train, y_train, parameters_grid, sinkhorn_length):
    """Train a CatBoost model. This performs cross validation as well.
    For now can only train a single output model.
    """
    if y_train.shape[1] != 1:
        raise ValueError("For now can only train a single output model.")
    
    # Initialize CatBoostRegressor
    base_model = catboost.core.CatBoostRegressor(verbose = False)

    # Cross Validation
    grid_search = GridSearchCV(estimator = base_model, param_grid = parameters_grid, cv = 5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    best_model = catboost.core.CatBoostRegressor(**best_params, verbose = False)

    # Fit the best model
    best_model.fit(x_train, y_train)

    return best_model

def inferenceCatBoost(model, x_test):
    """Hard coded for a single output.
    """
    if type(model) != catboost.core.CatBoostRegressor:
        raise TypeError("The model given is not a catboost from CatBoost package.")
    
    predictions = model.predict(x_test)

    return np.array(predictions).reshape(-1, 1)