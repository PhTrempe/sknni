from __future__ import absolute_import
from __future__ import division

import numbers

import numpy as np
from scipy.spatial.ckdtree import cKDTree


EPS = np.finfo(np.float32).eps
MIN_LAT = -90.0
MAX_LAT = 90.0 - EPS
MIN_LNG = -180.0
MAX_LNG = 180.0 - EPS
EARTH_MEAN_RADIUS = 6371.01


class SkNNI:
    """
    Spherical k-nearest neighbors interpolator.
    """

    def __init__(self, observations, r=EARTH_MEAN_RADIUS):
        """
        Initializes a SkNNI.

        :param observations: Array-like of observation triplets (lat, lng, val).
        :param r: Radius of the interpolation sphere (defaults to Earth's mean
            radius).
        """

        # Converts the observations array-like into a NumPy array
        observations = np.array(observations).astype(np.float32)

        if observations.ndim != 2 or observations.shape[-1] != 3:
            raise ValueError('Parameter "observations" must be a NumPy ndarray '
                             'of shape (-1, 3).')

        if np.isnan(observations).any():
            raise ValueError('Parameter "observations" contains at least one '
                             'NaN value.')

        # Clips the latitude and longitude values to their valid range
        observations[:, 0] = np.clip(observations[:, 0], MIN_LAT, MAX_LAT)
        observations[:, 1] = np.clip(observations[:, 1], MIN_LNG, MAX_LNG)

        if not isinstance(r, numbers.Real) or r <= 0:
            raise ValueError('Parameter "r" must be a strictly positive real '
                             'number.')

        self.observations = observations
        self.r = r

        # Converts degrees to radians
        self.obs_lats_rad = np.radians(observations[:, 0])
        self.obs_lngs_rad = np.radians(observations[:, 1])
        self.obs_values = observations[:, 2]

        # Converts polar coordinates to cartesian coordinates
        x, y, z = self.__polar_to_cartesian(
            self.obs_lats_rad, self.obs_lngs_rad, r)

        # Builds a k-dimensional tree using the transformed observations
        self.kd_tree = cKDTree(np.stack((x, y, z), axis=-1))

    def __call__(self, interp_coords, k=20, interp_fn=None):
        """
        Runs SkNNI for the given interpolation coordinates.

        :param interp_coords: Array-like of interpolation pairs (lat, lng).
        :param k: Number of nearest neighbors to consider (defaults to 20).
        :param interp_fn: Interpolation function (defaults to NDDNISD).

        :return: Interpolation triplets (lat, lng, interp_val).
        """

        # Converts the interp_coords array-like into a NumPy array
        interp_coords = np.array(interp_coords).astype(np.float32)

        if interp_coords.ndim != 2 or interp_coords.shape[-1] != 2:
            raise ValueError('Parameter "interp_coords" must be a NumPy '
                             'ndarray of shape (-1, 2).')

        if np.isnan(interp_coords).any():
            raise ValueError('Parameter "interp_coords" contains at least one '
                             'NaN value.')

        # Clips the latitude and longitude values to their valid range
        interp_coords[:, 0] = np.clip(interp_coords[:, 0], MIN_LAT, MAX_LAT)
        interp_coords[:, 1] = np.clip(interp_coords[:, 1], MIN_LNG, MAX_LNG)

        if not isinstance(k, numbers.Integral) or k <= 0:
            raise ValueError('Parameter k must be a strictly positive '
                             'integral number.')
        k = min(k, len(self.observations))

        if interp_fn is None:
            interp_fn = SkNNI.__nddnisd_interp_fn
        elif not callable(interp_fn):
            raise ValueError('Parameter interp_fn must be a callable '
                             'function-like object.')

        interp_lats_deg = interp_coords[:, 0]
        interp_lngs_deg = interp_coords[:, 1]

        # Converts degrees to radians
        interp_lats_rad = np.radians(interp_lats_deg)
        interp_lngs_rad = np.radians(interp_lngs_deg)

        # Converts polar coordinates to cartesian coordinates
        x, y, z = self.__polar_to_cartesian(
            interp_lats_rad, interp_lngs_rad, self.r)

        # Build a query for the spatial index
        kd_tree_query = np.stack((x, y, z), axis=-1)

        # Query the spatial index for the indices of the k nearest neighbors.
        _, knn_indices = self.kd_tree.query(kd_tree_query, k=k, n_jobs=-1)

        # Get lat, lng and value information for the k nearest neighbors.
        knn_lats = self.obs_lats_rad[knn_indices]
        knn_lngs = self.obs_lngs_rad[knn_indices]
        knn_values = self.obs_values[knn_indices]
        if knn_values.ndim < 2:
            knn_values = np.expand_dims(knn_values, axis=-1)

        # Get interpolation point's lat and lng for each k nearest neighbor.
        p_lats = np.tile(interp_lats_rad, (k, 1)).T
        p_lngs = np.tile(interp_lngs_rad, (k, 1)).T

        # Interpolate data values using the given interpolation function.
        interp_values = interp_fn(knn_lats, knn_lngs, knn_values,
                                  p_lats, p_lngs, self.r, k)

        return np.stack(
            (interp_lats_deg, interp_lngs_deg, interp_values), axis=-1)

    @staticmethod
    def __polar_to_cartesian(lat, lng, r):
        """
        Converts (lat, lng) coordinates in radians to cartesian coordinates.

        Args:
            lat: Latitude in radians.
            lng: Longitude in radians.
            r: Radius of the sphere.

        Returns: Cartesian coordinates from the given (lat, lng) coordinates.
        """

        x = r * np.cos(lng) * np.sin(lat)
        y = r * np.sin(lat) * np.sin(lng)
        z = r * np.cos(lat)
        return x, y, z

    @staticmethod
    def __great_circle_distance(a_lat, a_lng, b_lat, b_lng, r):
        """
        Calculates great circle distances in between (lat, lng) coordinates.

        Args:
            a_lat: Latitude of point A in radians.
            a_lng: Longitude of point A in radians.
            b_lat: Latitude of point B in radians.
            b_lng: Longitude of point B in radians.
            r: Radius of the sphere.

        Returns: Great circle distances.
        """

        a = np.square(np.sin((b_lat - a_lat) / 2)) + np.cos(a_lat) * \
            np.cos(b_lat) * np.square(np.sin((b_lng - a_lng) / 2))
        a = np.clip(a, 0, 1)
        return 2 * r * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    @staticmethod
    def __nddnisd_interp_fn(knn_lats, knn_lngs, knn_values, interp_lats,
                            interp_lngs, r, k):
        """
        Computes Neighborhood Distribution Debiased Normalised Inserse
        Squared Distance (NDDNISD) interpolation.

        Args:
            knn_lats: Latitude of the k nearest observation neighbors.
            knn_lngs: Longitude of the k nearest observation neighbors.
            knn_values: Value of the k nearest observation neighbors.
            interp_lats: Interpolation latitudes.
            interp_lngs: Interpolation longitudes.
            r: Radius of the sphere.
            k: Number of nearest neighbors to consider.

        Returns: Interpolated value for each pair of interpolation coordinates.
        """
        # For each interoplation point P_i = (interp_lat, interp_lng), we
        # compute the great-circle distance in between P_i and each of its k
        # nearest neighbors. So, we end up with a matrix which has
        # num_interp_points rows and k columns, and the data it contains is
        # the great-circle distance in between each interpolation point and
        # each of its k nearest neighboring observation points.
        gc_distance_p_knn = SkNNI.__great_circle_distance(
            interp_lats, interp_lngs, knn_lats, knn_lngs, r)

        # We compute weights based on great-circle distance.
        distance_weights = 1 / (np.square(gc_distance_p_knn) + EPS)
        distance_weights /= distance_weights.sum(axis=-1, keepdims=True)

        # For each interoplation point P_i = (interp_lat, interp_lng), we
        # compute the neighborhood's centroid. We then compute the distance
        # in between each neighbor and its neighborhood's centroid.
        centroids_lat = np.mean(knn_lats, axis=0)
        centroids_lng = np.mean(knn_lngs, axis=0)
        gc_distance_centroid_knn = SkNNI.__great_circle_distance(
            centroids_lat, centroids_lng, knn_lats, knn_lngs, r)

        # We compute weights based on distance weights by applying
        # neighborhood distribution debiasing.
        ndd_distance_weights = distance_weights * gc_distance_centroid_knn + EPS
        ndd_distance_weights /= ndd_distance_weights.sum(
            axis=-1, keepdims=True)

        # We compute the interpolation values based on k nearest neighbors'
        # observation values and calculated weights.
        return np.sum(ndd_distance_weights * knn_values, axis=-1)
