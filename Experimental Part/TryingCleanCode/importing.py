from packages import *

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

def r37_scalars(ids, test:bool, path: str = None):
    """From the Rotor37 dataset, imports the scalars of the observations ids.

    Args:
        ids(list): list of int.
        test(bool): A boolean that states if the scalars are for the test set or not (because the retrieving os different)
        path(str): add to the path.

    Returns:
        np.array: Array of the two input scalars.
        np.array: Array of the three output scalars.
    """
    if not test:
        padded_split = [str(i).zfill(9) for i in ids]
        efficiency = []
        omega = []
        P = []
        compression_ratio = []
        massflow = [] 
        for id in padded_split:
            coefficient_file_path = path+f'Rotor37/dataset/samples/sample_{id}/scalars.csv'
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

def data_for_regression(ids, path:str, test:bool, path_to_rotor37:str, path_to_metadata:str = None):
    """For the Rotor37 dataset, imports the data necessary for the regression task.

    Args:
        ids(list): list of int.
        path(str): Path to the saved Sinkhorn potentials.
        test(bool): If the data is for test or not.
        path_to_rotor_37: Path to the saved dataset.
        path_to_metadata: Path to the saved metadata.

    Returns:
        np.array: The array of the saved Sinkhorn potentials
        np.array: The array of the input scalars
    """
    # Import Sinkhorn Potentials
    if not path_to_metadata == None:
        with open(path_to_metadata) as f:
            metadata = json.load(f)
    
    sinkhorn_potentials = np.load(path)
    
    # Import scalars
    input_scalars, output_scalars = r37_scalars(ids = ids, test = test, path = path_to_rotor37)
    
    return sinkhorn_potentials, input_scalars, output_scalars, metadata



from pathlib import Path

def load_data_fn(split: str, directory:str ):

    meshes_file = directory / f"blade_meshes_{split}.h5"
    f = h5py.File(meshes_file, "r")
    X_points = np.array(f["points"])
    X_faces = np.array(f["faces"])
    Y_fields = np.array(f["output_fields"])
    f.close()

    scalars_file = directory / f"scalars_{split}.h5"
    f = h5py.File(scalars_file, "r")
    x_scalars = np.array(f["input_scalars"])
    y_scalars = np.array(f["output_scalars"])
    f.close()

    data = {
        "X_points": X_points,
        "Y_fields": Y_fields,
        "X_faces": X_faces,
        "x_scalars": x_scalars,
        "y_scalars": y_scalars,
        "x_scalars_names": ["omega", "pressure"],
        "y_scalars_names": [
            "inlet_massflow",
            "outlet_massflow",
            "compression_rate",
            "isentropic_efficiency",
            "polyentropic_efficiency",
        ],
    }
    return data
