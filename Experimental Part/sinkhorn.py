# This file helps compute the reference measure for the Sinkhorn algorithm.

import numpy as np

# Reference measure disk or sphere.
def sample_points_on_sphere(num_points, center, radius):
    theta = np.random.default_rng().uniform(0, np.pi, num_points)
    phi = np.random.default_rng().uniform(0, 2 * np.pi, num_points)

    # Calculate Cartesian coordinates
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    x = center[0] + radius * sin_theta * cos_phi
    y = center[1] + radius * sin_theta * sin_phi
    z = center[2] + radius * cos_theta

    return np.column_stack((x, y, z))

def uniform_sample_points_on_sphere(num_points, center, radius):
    # Generate evenly spaced points on a unit sphere using spherical coordinates
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_points)
    theta = np.pi * (1 + 5**0.5) * indices

    # Convert spherical coordinates to Cartesian coordinates
    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    return np.column_stack((x, y, z))

def compute_center_of_blades(blades):
  # Reshape the blades to have a single array with shape (num_blade * blade_size, 3)
  combined_blades = np.concatenate(blades)
  # Compute the mean center
  mean_center = np.mean(combined_blades, axis=0)
  # Compute the distances from each point to the mean center
  distances = np.linalg.norm(combined_blades - mean_center, axis=1)
  # Compute the radius of the smallest sphere
  radius = np.max(distances)

  return mean_center, radius

def define_reference_measure(data, ref_measure: str, ref_measure_size):
    """Defines the reference measure to use. 
    None of those reference measures are data driven for now.
    Args:

    Returns:
        WeightedPointCloud object.
    """
    if ref_measure == "blade0":
      ref_measure_cloud = data[0, :, :]
    elif ref_measure == "sphere":
      center, radius = compute_center_of_blades(data)
      ref_measure_cloud = sample_points_on_sphere(num_points = ref_measure_size, radius = radius, center = center)
    elif ref_measure == "UniformSphere":
      center, radius = compute_center_of_blades(data)
      ref_measure_cloud = uniform_sample_points_on_sphere(num_points = ref_measure_size, radius = radius, center = center)
    elif ref_measure == "UniformSphereFar":
          _, radius = compute_center_of_blades(data)
          ref_measure_cloud = uniform_sample_points_on_sphere(num_points = ref_measure_size, radius = radius, center = np.array([0, 0, 0]))
    elif ref_measure == "disk":
      center, radius = compute_center_of_blades(data)
      ref_measure_cloud = sample_points_on_sphere(num_points = ref_measure_size, radius = radius, center = center)
      ref_measure_cloud[:, 2] = ref_measure_cloud[:, 2]*0
    elif ref_measure == "random":
      ref_measure_cloud = data[np.random.randint(0, len(data)), :, :]
    else:
      raise ValueError("The reference measure name must be one of: blade0, disk, sphere, random.")

    return ref_measure_cloud