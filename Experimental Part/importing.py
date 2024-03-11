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

def blades(split: list, sample_size:int, path:str = None, sample_fn = None) -> list:
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

def train_data(problem, problem_txt, test, 
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


def test_data(problem, problem_txt, test, 
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
    #efficiency = []
    omega = []
    P = []
    #compression_ratio = []
    #massflow = [] 
    for id in padded_split:
        coefficient_file_path = f'Rotor37/dataset/samples/sample_{id}/scalars.csv'
        scalars = pd.read_csv(coefficient_file_path)
        ## Adding to our data
        #efficiency.append(scalars["Efficiency"][0])
        omega.append(scalars["Omega"][0])
        P.append(scalars["P"][0])
        #compression_ratio.append(scalars["Compression_ratio"][0])
        #massflow.append(scalars["Massflow"][0])
    with h5py.File('Rotor37/scalars_test.h5', 'r') as file:
        output_scalars = np.array(file['output_scalars'])
        massflow = np.mean(output_scalars[:, :2], axis=1).tolist()
        compression_ratio = output_scalars[:, 2].tolist()
        efficiency = output_scalars[:, 3].tolist()
        
        input_scalars = np.array(file['input_scalars'])
        omega2 = input_scalars[:, 0].tolist()
        P2 = input_scalars[:, 1].tolist()
        if not (omega2 == omega) or (P2 == P):
            raise ImportError("The test values are not in the same order as the Sinkhorn potentials.")
    
    return sinkhorn_potentials, efficiency, omega, P, compression_ratio, massflow, metadata

