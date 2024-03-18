from packages import *

@struct.dataclass
class WeightedPointCloud:
  """A weighted point cloud.
  
  Attributes:
    cloud: Array of shape (n, d) where n is the number of points and d the dimension.
    weights: Array of shape (n,) where n is the number of points.
  """
  cloud: jnp.array
  weights: jnp.array

  def __len__(self):
    return self.cloud.shape[0]


@struct.dataclass
class VectorizedWeightedPointCloud:
  """Vectorized version of WeightedPointCloud.

  Assume that b clouds are all of size n and dimension d.
  
  Attributes:
    _private_cloud: Array of shape (b, n, d) where n is the number of points and d the dimension.
    _private_weights: Array of shape (b, n) where n is the number of points.
  
  Methods:
    unpack: returns the cloud and weights.
  """
  _private_cloud: jnp.array
  _private_weights: jnp.array

  def __getitem__(self, idx):
    return WeightedPointCloud(self._private_cloud[idx], self._private_cloud[idx])
  
  def __len__(self):
    return self._private_cloud.shape[0]
  
  def __iter__(self):
    for i in range(len(self)):
      yield self[i]

  def unpack(self):
    return self._private_cloud, self._private_weights

def pad_point_cloud(point_cloud, max_cloud_size, fail_on_too_big=True):
  """Pad a single point cloud with zeros to have the same size.
  
  Args:
    point_cloud: a weighted point cloud.
    max_cloud_size: the size of the biggest point cloud.
    fail_on_too_big: if True, raise an error if the cloud is too big for padding.
  
  Returns:
    a WeightedPointCloud with padded cloud and weights.
  """
  cloud, weights = point_cloud.cloud, point_cloud.weights
  delta = max_cloud_size - cloud.shape[0]
  if delta <= 0:
    if fail_on_too_big:
      assert False, 'Cloud is too big for padding.'
    return point_cloud

  ratio = 1e-3  # less than 0.1% of the total mass.
  smallest_weight = jnp.min(weights) / delta * ratio
  small_weights = jnp.ones(delta) * smallest_weight

  weights = weights * (1 - ratio)  # keep 99.9% of the mass.
  weights = jnp.concatenate([weights, small_weights], axis=0)

  cloud = jnp.pad(cloud, pad_width=((0, delta), (0,0)), mode='mean')

  point_cloud = WeightedPointCloud(cloud, weights)

  return point_cloud

def pad_point_clouds(cloud_list):
  """Pad the point clouds with zeros to have the same size.

  Note: this function should be used outside of jax.jit because the computation graph
        is huge. O(len(cloud_list)) nodes are generated.

  Args:
    cloud_list: a list of WeightedPointCloud.
  
  Returns:
    a VectrorizedWeightedPointCloud with padded clouds and weights.
  """
  # sentinel for unified processing of all clouds, including biggest one.
  max_cloud_size = max([len(cloud) for cloud in cloud_list]) + 1
  sentinel_padder = partial(pad_point_cloud, max_cloud_size=max_cloud_size)

  cloud_list = list(map(sentinel_padder, cloud_list))
  coordinates = jnp.stack([cloud.cloud for cloud in cloud_list])
  weights = jnp.stack([cloud.weights for cloud in cloud_list])
  return VectorizedWeightedPointCloud(coordinates, weights)

def clouds_barycenter(points):
  """Compute the barycenter of a set of clouds.
  
  Args:
    points: a VectorizedWeightedPointCloud.
    
  Returns:
    a barycenter of the clouds of points, of shape (1, d) where d is the dimension.
  """
  clouds, weights = points.unpack()
  barycenter = jnp.sum(clouds * weights[:,:,jnp.newaxis], axis=1)
  barycenter = jnp.mean(barycenter, axis=0, keepdims=True)
  return barycenter


def to_simplex(mu):
  """Project weights to the simplex.
  
  Args: 
    mu: a WeightedPointCloud.
    
  Returns:
    a WeightedPointCloud with weights projected to the simplex."""
  if mu.weights is None:
    mu_weights = None
  else:
    mu_weights = jax.nn.softmax(mu.weights)
  return replace(mu, weights=mu_weights)


def reparametrize_mu(mu, cloud_barycenter, scale):
  """Re-parametrize mu to be invariant by translation and scaling.

  Args:
    mu: a WeightedPointCloud.
    cloud_barycenter: Array of shape (1, d) where d is the dimension.
    scale: float, scaling parameter for the re-parametrization of mu.
  
  Returns:
    a WeightedPointCloud with re-parametrized weights and cloud.
  """
  # invariance by translation : recenter mu around its mean
  mu_cloud = mu.cloud - jnp.mean(mu.cloud, axis=0, keepdims=True)  # center.
  mu_cloud = scale * jnp.tanh(mu_cloud)  # re-parametrization of the domain.
  mu_cloud = mu_cloud + cloud_barycenter  # re-center toward barycenter of all clouds.
  return replace(mu, cloud=mu_cloud)

def clouds_to_dual_sinkhorn(points, 
                            mu, 
                            init_dual=(None, None),
                            scale=1.,
                            has_aux=False,
                            sinkhorn_solver_kwargs=None, 
                            parallel: bool = True,
                            batch_size: int = -1,
                            numerator: int = 1,
                            denominator: int = 1):
  """Compute the embeddings of the clouds with regularized OT towards mu.
  
  Args:
    points: a VectorizedWeightedPointCloud.
    init_dual: tuple of two arrays of shape (b, n) and (b, m) where b is the number of clouds,
               n is the number of points in each cloud, and m the number of points in mu.
    scale: float, scaling parameter for the re-parametrization of mu.
    has_aux: bool, whether to return the full Sinkhorn output or only the dual variables.
    sinkhorn_solver_kwargs: dict, kwargs for the Sinkhorn solver.
      Must contain the key 'epsilon' for the regularization parameter.

  Returns:
    a tuple (dual, init_dual) with dual variables of shape (n, m) where n is the number of points
    and m the number of points in mu, and init_dual a tuple (init_dual_cloud, init_dual_mu) 
  """
  sinkhorn_epsilon = sinkhorn_solver_kwargs.pop('epsilon')
  
  # weight projection
  barycenter = clouds_barycenter(points)
  mu = to_simplex(mu)

  # cloud projection
  mu = reparametrize_mu(mu, barycenter, scale)

  def sinkhorn_single_cloud(cloud, weights, init_dual):
    geom = PointCloud(cloud, mu.cloud,
                      epsilon=sinkhorn_epsilon)
    ot_prob = LinearProblem(geom,
                            weights,
                            mu.weights)
    solver = Sinkhorn(**sinkhorn_solver_kwargs)
    ot = solver(ot_prob, init=init_dual)
    return ot
  
  if parallel:
    if batch_size == -1:
        parallel_sinkhorn = jax.vmap(sinkhorn_single_cloud,
                                    in_axes=(0, 0, (0, 0)),
                                    out_axes=0)
        outs = parallel_sinkhorn(*points.unpack(), init_dual)
        return outs.g
    else:
      raise ValueError("Not coded yet") 
  else:
    list_of_g_potentials = []
    clouds, weights = points.unpack()
    sinkhorn_single_cloud = jax.jit(sinkhorn_single_cloud)

    start_idx = (numerator - 1) * len(clouds) // denominator
    end_idx = numerator * len(clouds) // denominator

    clouds = clouds[start_idx:end_idx]
    weights = weights[start_idx:end_idx]

    for i in range(len(clouds)):
      ot_problem = sinkhorn_single_cloud(clouds[i], weights[i], init_dual)
      list_of_g_potentials.append(ot_problem.g)
    g_potentials_array = jnp.stack(list_of_g_potentials)
    return g_potentials_array

def proceed(data: np.ndarray, epsilon, ref_measure, numerateur: int, denominateur: int ) -> np.ndarray:
    """For an array of clouds, perform Sinkhorn algorithm against the reference measure.

    Args:

    Returns:
    """
    ## First we convert the list all the sampled distributions to WeightedPointCloud objects
    list_of_weighted_point_clouds = []
    for sample in data:
        distrib_cloud = WeightedPointCloud(
            cloud=jnp.array(sample),
            weights=jnp.ones(len(sample)))
        list_of_weighted_point_clouds.append(distrib_cloud)

    ## We need to convert the cloud list to a VectorizedWeightedPointCloud
    x_cloud = pad_point_clouds(list_of_weighted_point_clouds)

    ## We choose our epsilon parameter and perform the sinkhirn algorithm
    sinkhorn_solver_kwargs = {'epsilon': epsilon}
    # Train
    sinkhorn_potentials = clouds_to_dual_sinkhorn(points = x_cloud, mu = ref_measure, 
                                                    sinkhorn_solver_kwargs = sinkhorn_solver_kwargs, 
                                                    parallel = False, # going into the for loop
                                                    batch_size = -1,
                                                    numerator = numerateur,
                                                    denominator = denominateur)
    
    return np.array(sinkhorn_potentials)


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



def define_reference_measure(data, ref_measure: str, ref_measure_size) -> WeightedPointCloud:
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
    elif ref_measure == "disk":
      center, radius = compute_center_of_blades(data)
      ref_measure_cloud = sample_points_on_sphere(num_points = ref_measure_size, radius = radius, center = center)
      ref_measure_cloud[:, 2] = ref_measure_cloud[:, 2]*0
    elif ref_measure == "random":
      ref_measure_cloud = data[np.random.randint(0, len(data)), :, :]
    else:
      raise ValueError("The reference measure name must be one of: blade0, disk, sphere, random.")
    
    ref_measure_cloud = WeightedPointCloud(
       cloud=jnp.array(ref_measure_cloud),
       weights=jnp.ones(len(ref_measure_cloud))
    )

    return ref_measure_cloud