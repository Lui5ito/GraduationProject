from packages import *

#################
###--GENERAL--###
#################

def train(name: str, x_train, y_train, parameters_grid, sinkhorn_length):
    if name == "krr":
        krr_model, krr_parameters = trainKRR(x_train, y_train, parameters_grid, sinkhorn_length)
        return krr_model, krr_parameters
    if name == "gp":
        gp_model, gp_parameters = trainGP(x_train, y_train, sinkhorn_length)
        return gp_model, gp_parameters
    if name == "catboost":
        catboost_model, catboost_parameters = trainCatBoost(x_train, y_train, parameters_grid)
        return catboost_model, catboost_parameters
    else:
        raise ValueError(f"The model {name} asked has not been coded yet.")
    
def inference(name, model, x_test, models_parameters, normalizer):
    if name == "krr":
        krr_predictions = inferenceKRR(model = model, x_test = x_test, models_parameters = models_parameters, normalizer = normalizer)
        return krr_predictions
    if name == "gp":
        gp_predictions = inferenceGP(model = model, x_test = x_test)
        return gp_predictions
    if name == "catboost":
        catboost_predictions = inferenceCatBoost(model = model, x_test = x_test)
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

    alpha = best_model.alpha
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

    krr_parameters = {"regularization": alpha,
                      "first_kernel": gamma,
                      "second_kernel": [gamma1, gamma2],
                      "sinkhorn_train": sinkhorn_train,
                      "scalars_train": scalars_train, 
                      "sinkhorn_length": sinkhorn_length}

    return best_model, krr_parameters

def inferenceKRR(model, x_test, models_parameters:dict, normalizer):
    """For a given krr, predicts.
    """
    if type(model) != sklearn.kernel_ridge.KernelRidge:
        raise TypeError("The model given is not a kernel ridge regression from sklearn.")
    
    kernel_sinkhorn = kernels.RBF(length_scale=np.array([models_parameters["first_kernel"]]))
    kernel_scalars = kernels.RBF(length_scale=np.array(models_parameters["second_kernel"]))

    sinkhorn_test = x_test[:, 0:models_parameters["sinkhorn_length"]]
    scalars_test = x_test[:, models_parameters["sinkhorn_length"]:]
    scalars_test = normalizer.transform(scalars_test)

    kernel_matrix_sinkhorn_test = kernel_sinkhorn(sinkhorn_test, models_parameters["sinkhorn_train"])
    kernel_matrix_scalars_test = kernel_scalars(scalars_test, models_parameters["scalars_train"])
    k_test = kernel_matrix_sinkhorn_test * kernel_matrix_scalars_test

    predictions = model.predict(k_test)

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

    model_parameters = {"kernel": model.kern}

    return model, model_parameters

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

def trainCatBoost(x_train, y_train, parameters_grid):
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

    model_parameters = best_params

    return best_model, model_parameters

def inferenceCatBoost(model, x_test):
    """Hard coded for a single output.
    """
    if type(model) != catboost.core.CatBoostRegressor:
        raise TypeError("The model given is not a catboost from CatBoost package.")
    
    predictions = model.predict(x_test)

    return np.array(predictions).reshape(-1, 1)