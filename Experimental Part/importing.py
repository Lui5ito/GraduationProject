# This file is helds the functions that are made to import the data from the Rotor37 dataset. Those functions are used throughout the notebooks and files.


import numpy as np
import h5py
import pandas as pd
import json

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

def r37_blades(ids: list, path:str = None,) -> np.array:
    """From the Rotor37 dataset, imports the blades of observations ids.

    Args:
        ids(list): list of int.
        path(str): add to the path.

    Returns:
        np.array: An array of all blades.
    """

    padded_split = [str(i).zfill(9) for i in ids]

    distributions = []

    for id in padded_split:
        ## File paths Personal Computer
        cgns_file_path = f'Rotor37/dataset/samples/sample_{id}/meshes/mesh_000000000.cgns'
        if path is not None:
            cgns_file_path = path + cgns_file_path
        ## Computing the coordinates
        x, y, z = read_cgns_coordinates(cgns_file_path)
        blade = np.column_stack((x, y, z))
        ## Adding to our data
        distributions.append(blade)
    
    return np.array(distributions)

def r37_scalars(ids, test:bool = False) -> tuple:
    """From the Rotor37 dataset, imports the scalars of the observations ids.

    Args:
        ids(list): list of int.
        test(bool): A boolean that states if the scalars are for the test set or not (because the retrieving os different)
        path(str): add to the path.

    Returns:
        tuple: Tuple of two np.array, one of the two input scalars and the other of the three output scalars.
    """
    # If the split is not test:
    if not test:
        padded_split = [str(i).zfill(9) for i in ids]
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
        
        omega = np.array(omega).reshape(-1, 1)
        P = np.array(P).reshape(-1, 1)
        efficiency = np.array(efficiency).reshape(-1, 1)
        massflow = np.array(massflow).reshape(-1, 1)
        compression_ratio = np.array(compression_ratio).reshape(-1, 1)
    # If the split is test, we need a different way to import the scalars
    else:
        padded_split = [str(i).zfill(9) for i in ids]
        # Import scalars
        omega = []
        P = []
        for id in padded_split:
            coefficient_file_path = f'Rotor37/dataset/samples/sample_{id}/scalars.csv'
            scalars = pd.read_csv(coefficient_file_path)
            omega.append(scalars["Omega"][0])
            P.append(scalars["P"][0])
        omega = np.array(omega).reshape(-1, 1)
        P = np.array(P).reshape(-1, 1)
        with h5py.File('Rotor37/scalars_test.h5', 'r') as file:
            output_scalars = np.array(file['output_scalars'])
            massflow = np.mean(output_scalars[:, :2], axis=1).reshape(-1, 1)
            compression_ratio = output_scalars[:, 2].reshape(-1, 1)
            efficiency = output_scalars[:, 3].reshape(-1, 1)

    return np.hstack([omega, P]), np.hstack([efficiency, massflow, compression_ratio])   
