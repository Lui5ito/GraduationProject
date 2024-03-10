from packages import *


def train_data(sinkhorn_potentials, list_of_X, list_of_Y):
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

def test_data(sinkhorn_potentials, list_of_X, list_of_Y, train_normalizer):
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