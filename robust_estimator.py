import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from typing import Tuple, Optional
import warnings
import os
import time

from train_HMC import *

class RobustMeanEstimator:
    """
    Robust mean estimation for high-dimensional data with corruption.
    
    Uses efficient spectral filtering to remove outliers that inflate
    the principal eigenvalues of the covariance matrix.
    """
    
    def __init__(self, corruption_level: float = 0.1, max_iterations: int = 5):
        """
        Parameters
        ----------
        corruption_level : float
            Expected fraction of corrupted samples (ε)
        max_iterations : int
            Maximum number of spectral filtering iterations
        """
        self.epsilon = corruption_level
        self.max_iterations = max_iterations
        self.fitted_mean_ = None
        self.robust_covariance_ = None
        self.inlier_mask_ = None
        self.n_projections = 100  # Number of random projections for depth computation
        
    def spectral_filtering_step(self, X: np.ndarray, mean: np.ndarray,
                                threshold_factor: float = 2.5) -> np.ndarray:
        """
        One step of spectral filtering: remove outliers along principal direction.
        
        Algorithm:
        1. Compute covariance matrix
        2. Find direction of maximum variance (top eigenvector)
        3. Remove points with extreme projections in this direction
        
        This is MUCH faster than depth-based methods: O(nd²) vs O(n²d × projections)
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        mean : array, shape (n_features,)
            Current mean estimate
        threshold_factor : float
            Remove points beyond threshold_factor × std in principal direction
            
        Returns
        -------
        mask : array, shape (n_samples,), dtype=bool
            True for inliers, False for outliers
        """
        n, d = X.shape
        
        # Center the data
        X_centered = X - mean
        
        # Compute covariance (use more stable formulation for high-d)
        if n > d:
            # Standard covariance
            cov = (X_centered.T @ X_centered) / (n - 1)
        else:
            # Use Gram matrix trick for n < d (more efficient)
            gram = (X_centered @ X_centered.T) / (n - 1)
            # Don't need full eigendecomposition, just check spectral norm
            max_eigenvalue = np.linalg.norm(gram, ord=2)
            
            # Simplified filtering based on distance to mean
            distances = np.linalg.norm(X_centered, axis=1)
            median_dist = np.median(distances)
            threshold = threshold_factor * median_dist
            return distances < threshold
        
        # Find top eigenvector (direction of max variance)
        # Use eigh for symmetric matrices (faster than eig)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Get largest eigenvalue and its direction
        idx_max = np.argmax(eigenvalues)
        max_eigenvalue = eigenvalues[idx_max]
        principal_direction = eigenvectors[:, idx_max]
        
        # Expected variance: use median of eigenvalues (robust to outliers)
        median_eigenvalue = np.median(eigenvalues)
        
        # If max eigenvalue is much larger than median, we have outliers
        if max_eigenvalue > threshold_factor * median_eigenvalue:
            # Project data onto principal direction
            projections = X_centered @ principal_direction
            
            # Robust threshold using MAD (Median Absolute Deviation)
            median_proj = np.median(projections)
            mad = np.median(np.abs(projections - median_proj))
            # Convert MAD to std: std ≈ 1.4826 × MAD for Gaussian
            robust_std = 1.4826 * mad
            
            # Remove points beyond threshold in this direction
            threshold = threshold_factor * robust_std
            mask = np.abs(projections - median_proj) < threshold
            
            return mask
        else:
            # Variance is reasonable, no obvious outliers
            return np.ones(n, dtype=bool)
    
    def geometric_median(self, X: np.ndarray, weights: Optional[np.ndarray] = None,
                         max_iter: int = 100, tol: float = 1e-5) -> np.ndarray:
        """
        Compute weighted geometric median (L2 robust mean).
        
        The geometric median minimizes sum of L2 distances:
            argmin_μ Σ w_i ||X_i - μ||_2
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data points
        weights : array-like, shape (n_samples,), optional
            Sample weights (uniform if None)
        
        Returns
        -------
        median : array, shape (n_features,)
            Geometric median
        """
        n, d = X.shape
        if weights is None:
            weights = np.ones(n) / n
        else:
            weights = weights / weights.sum()
        
        # Initialize at weighted mean
        mu = (weights[:, None] * X).sum(axis=0)
        
        for iteration in range(max_iter):
            mu_old = mu.copy()
            
            # Compute distances
            dists = np.linalg.norm(X - mu, axis=1)
            dists[dists < 1e-10] = 1e-10  # Avoid division by zero
            
            # Weiszfeld's algorithm update
            inv_dists = weights / dists
            mu = (inv_dists[:, None] * X).sum(axis=0) / inv_dists.sum()
            
            # Check convergence
            if np.linalg.norm(mu - mu_old) < tol:
                break
                
        return mu
    
    def trimmed_mean(self, X: np.ndarray, trim_fraction: float = 0.1) -> np.ndarray:
        """
        Compute trimmed mean by removing extreme values.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data points
        trim_fraction : float
            Fraction of most extreme points to remove from each dimension
            
        Returns
        -------
        trimmed_mean : array, shape (n_features,)
            Trimmed mean
        """
        n, d = X.shape
        trim_count = int(trim_fraction * n)
        
        if trim_count >= n // 2:
            # Too much trimming, fall back to median
            return np.median(X, axis=0)
        
        # For each dimension, remove the most extreme values
        trimmed_means = []
        for j in range(d):
            values = X[:, j]
            # Sort and remove trim_count from each end
            sorted_values = np.sort(values)
            if trim_count > 0:
                trimmed_values = sorted_values[trim_count:-trim_count]
            else:
                trimmed_values = sorted_values
            trimmed_means.append(np.mean(trimmed_values))
        
        return np.array(trimmed_means)
    
    def compute_projection_depth(self, X: np.ndarray, point: np.ndarray, n_directions: int = 100) -> float:
        """
        Compute projection depth of a point with respect to a dataset.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Reference dataset
        point : array, shape (n_features,)
            Point to compute depth for
        n_directions : int
            Number of random projection directions
            
        Returns
        -------
        depth : float
            Projection depth (between 0 and 1)
        """
        n, d = X.shape
        if n <= 1:
            return 1.0
        
        # Generate random unit directions
        np.random.seed(42)  # For reproducibility
        directions = np.random.randn(n_directions, d)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        
        depths = []
        for direction in directions:
            # Project data and point onto direction
            projections = X @ direction
            point_proj = point @ direction
            
            # Compute rank of point among projections
            rank = np.sum(projections <= point_proj)
            depth_1d = min(rank, n - rank) / n
            depths.append(depth_1d)
        
        return min(depths)  # Tukey depth is minimum over all directions
    
    def iterative_spectral_filtering(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iteratively filter outliers using spectral method and recompute robust mean.
        
        Much faster than depth-based approach:
        - Depth-based: O(n² × d × n_projections) per round
        - Spectral: O(n × d²) per round (or O(n × d) for n < d)
        
        Algorithm:
        1. Compute geometric median (or mean for speed)
        2. Spectral filtering: remove outliers along principal direction
        3. Repeat on filtered data
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            
        Returns
        -------
        robust_mean : array, shape (n_features,)
        inlier_mask : array, shape (n_samples,), dtype=bool
        """
        n, d = X.shape
        mask = np.ones(n, dtype=bool)
        
        # Adaptive threshold based on corruption level
        threshold_factor = 2.0 + 3.0 * self.epsilon  # More aggressive for higher ε
        
        for iteration in range(self.max_iterations):
            X_filtered = X[mask]
            n_filtered = len(X_filtered)
            
            # Safety: don't remove too many points
            if n_filtered < (1 - 2 * self.epsilon) * n:
                warnings.warn(f"Filtered too aggressively ({n_filtered}/{n} remaining), stopping.")
                break
            
            # Compute mean (use simple mean for speed, or geometric median for robustness)
            if d < 100:  # For low-d, geometric median is fast enough
                current_mean = self.geometric_median(X_filtered, max_iter=50)
            else:  # For high-d, use trimmed mean (faster)
                current_mean = self.trimmed_mean(X_filtered, trim_fraction=self.epsilon)
            
            # Spectral filtering step
            filter_mask = self.spectral_filtering_step(X_filtered, current_mean, 
                                                       threshold_factor=threshold_factor)
            
            # Update global mask
            temp_mask = mask.copy()
            temp_mask[mask] = filter_mask
            
            n_removed = mask.sum() - temp_mask.sum()
            
            # Check convergence
            if n_removed == 0:
                break
                
            mask = temp_mask
            
            # Print progress
            if iteration == 0 or n_removed > 0:
                print(f"  Iteration {iteration + 1}: removed {n_removed} outliers, "
                      f"{mask.sum()}/{n} remaining")
        
        # Final robust mean
        X_final = X[mask]
        if d < 100:
            final_mean = self.geometric_median(X_final, max_iter=100)
        else:
            final_mean = self.trimmed_mean(X_final, trim_fraction=0.05)
        
        return final_mean, mask

    
    def filter_by_depth(self, X: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """
        Filter points by Tukey depth around geometric median.
        
        Keep points with depth >= threshold (central points).
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        threshold : float
            Minimum depth to keep a point
            
        Returns
        -------
        mask : array, shape (n_samples,), dtype=bool
            True for points to keep
        """
        # Use geometric median as reference
        center = self.geometric_median(X)
        
        depths = np.array([
            self.compute_projection_depth(X, x, n_directions=self.n_projections)
            for x in X
        ])
        
        return depths >= threshold
    
    def iterative_filtering(self, X: np.ndarray, max_rounds: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Iteratively filter outliers and recompute robust mean.
        
        Algorithm:
        1. Compute geometric median
        2. Filter by projection depth
        3. Repeat on filtered data
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        max_rounds : int
            Maximum filtering iterations
            
        Returns
        -------
        robust_mean : array, shape (n_features,)
        inlier_mask : array, shape (n_samples,), dtype=bool
        """
        n = len(X)
        mask = np.ones(n, dtype=bool)
        
        # Adaptive threshold based on corruption level
        # depth ~ 0.5 - ε for true mean
        depth_threshold = max(0.5 - 2 * self.epsilon, 0.2)
        
        for round_idx in range(max_rounds):
            X_filtered = X[mask]
            
            if len(X_filtered) < 0.5 * n:
                warnings.warn(f"Filtered too many points ({len(X_filtered)}/{n}), stopping.")
                break
            
            # Compute depth-based filter
            depth_mask = self.filter_by_depth(X_filtered, threshold=depth_threshold)
            
            # Update global mask
            temp_mask = mask.copy()
            temp_mask[mask] = depth_mask
            
            # Check convergence
            if temp_mask.sum() == mask.sum():
                break
                
            mask = temp_mask
            
        return self.geometric_median(X[mask]), mask
    
    def fit(self, X: np.ndarray) -> 'RobustMeanEstimator':
        """
        Fit robust mean estimator using fast spectral filtering.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Training data (potentially corrupted)
            
        Returns
        -------
        self
        """
        X = np.asarray(X)
        n, d = X.shape
        
        print(f"Fitting robust mean on {n} samples, {d} features")
        
        # Use spectral filtering (much faster than depth-based)
        self.fitted_mean_, self.inlier_mask_ = self.iterative_spectral_filtering(X)
        
        print(f"Final: kept {self.inlier_mask_.sum()}/{n} samples "
              f"({100*self.inlier_mask_.mean():.1f}%)")
        
        # Robust covariance from inliers
        X_inliers = X[self.inlier_mask_]
        if len(X_inliers) > d:  # Need more samples than dimensions
            self.robust_covariance_ = np.cov(X_inliers.T)
        else:
            warnings.warn("Too few inliers for covariance estimation")
            self.robust_covariance_ = np.eye(d)
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Compute robust Mahalanobis distance to fitted mean.
        
        Useful for anomaly detection / outlier scoring.
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        
        Returns
        -------
        distances : array, shape (n_samples,)
        """
        if self.fitted_mean_ is None:
            raise ValueError("Must call fit() before predict()")
        
        X = np.asarray(X)
        
        # Try Mahalanobis distance, fall back to L2 if covariance is singular
        try:
            cov_inv = np.linalg.inv(self.robust_covariance_)
            diff = X - self.fitted_mean_
            distances = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))
        except np.linalg.LinAlgError:
            distances = np.linalg.norm(X - self.fitted_mean_, axis=1)
            
        return distances


class RobustPosteriorInference:
    """
    Robust posterior inference for emulator outputs.
    
    Optimized pipeline:
    1. Project high-dim images to feature space (PCA - efficient)
    2. Find nearest neighbors (KD-tree for speed)
    3. Robustly aggregate their parameters (spectral filtering)
    """
    
    def __init__(self, n_components: int = 50, corruption_level: float = 0.1):
        """
        Parameters
        ----------
        n_components : int
            Number of PCA components for dimensionality reduction
        corruption_level : float
            Expected corruption level
        """
        self.n_components = n_components
        self.epsilon = corruption_level
        
        # Components
        self.pca = PCA(n_components=n_components)
        self.robust_estimator = RobustMeanEstimator(corruption_level=corruption_level)
        
        # Fitted data
        self.features_train_ = None
        self.params_train_ = None
        self.kdtree_ = None  # For fast nearest neighbor search
        
    def fit(self, images: np.ndarray, params: np.ndarray):
        """
        Fit the robust posterior inference model.
        
        Optimized for large datasets:
        - Uses randomized PCA if n_samples is large
        - Builds KD-tree for fast neighbor search
        
        Parameters
        ----------
        images : array, shape (n_samples, height, width) or (n_samples, n_pixels)
            Training images (flatten if needed)
        params : array, shape (n_samples, n_params)
            Corresponding parameters (5 params: 2 model + 3 realization)
        """
        # Flatten images if needed
        if images.ndim == 3:
            n_samples = images.shape[0]
            images = images.reshape(n_samples, -1)
        
        n_samples, n_pixels = images.shape
        print(f"Training on {n_samples} images of dimension {n_pixels}")
        
        # For large datasets, use randomized PCA (much faster)
        if n_samples > 5000 or n_pixels > 10000:
            print("Using randomized PCA for speed...")
            from sklearn.decomposition import PCA as StandardPCA
            self.pca = StandardPCA(n_components=self.n_components, svd_solver='randomized')
        
        # Compute PCA projection
        print("Computing PCA projection...")
        import time
        t0 = time.time()
        self.features_train_ = self.pca.fit_transform(images)
        t1 = time.time()
        
        self.params_train_ = params
        
        print(f"PCA done in {t1-t0:.2f}s")
        print(f"Reduced to {self.n_components} dimensions")
        print(f"Explained variance: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Build KD-tree for fast nearest neighbor queries
        print("Building KD-tree for fast neighbor search...")
        from scipy.spatial import cKDTree
        self.kdtree_ = cKDTree(self.features_train_)
        print("KD-tree built")
        
        return self
    
    def infer_model_params(self, test_image: np.ndarray, k_neighbors: int = 50,
                          return_uncertainty: bool = True) -> dict:
        """
        Robustly infer model parameters from a test image.
        
        Optimized with KD-tree for fast neighbor search.
        
        Strategy:
        1. Project test image to feature space
        2. Find k nearest neighbors (fast KD-tree query)
        3. Robustly estimate mean of their model parameters (spectral filtering)
        
        Parameters
        ----------
        test_image : array, shape (height, width) or (n_pixels,)
            Test image (potentially noisy)
        k_neighbors : int
            Number of nearest neighbors to aggregate
        return_uncertainty : bool
            If True, also return posterior uncertainty estimates
            
        Returns
        -------
        results : dict
            'model_params': Robust estimate of 2 model parameters
            'model_params_std': Standard deviation (if return_uncertainty=True)
            'distances': Distances to k nearest neighbors
        """
        # Flatten if needed
        if test_image.ndim == 2:
            test_image = test_image.flatten()
        
        # Project to feature space
        test_features = self.pca.transform(test_image.reshape(1, -1))
        
        # Find k nearest neighbors using KD-tree (MUCH faster than naive search)
        # O(log n) per query instead of O(n)
        if self.kdtree_ is not None:
            distances, neighbor_indices = self.kdtree_.query(test_features[0], k=k_neighbors)
        else:
            # Fallback to naive search if KD-tree not available
            distances_all = np.linalg.norm(self.features_train_ - test_features, axis=1)
            neighbor_indices = np.argsort(distances_all)[:k_neighbors]
            distances = distances_all[neighbor_indices]
        
        # Extract model parameters (first 2 columns) from neighbors
        neighbor_params = self.params_train_[neighbor_indices, :2]
        
        # Robustly estimate mean using spectral filtering
        estimator = RobustMeanEstimator(corruption_level=self.epsilon, max_iterations=3)
        estimator.fit(neighbor_params)
        
        robust_mean = estimator.fitted_mean_
        
        results = {
            'model_params': robust_mean,
            'distances': distances
        }
        
        if return_uncertainty:
            # Use robust covariance for uncertainty
            if estimator.robust_covariance_ is not None:
                results['model_params_std'] = np.sqrt(np.diag(estimator.robust_covariance_))
            else:
                results['model_params_std'] = None
                
        return results
    
    def infer_batch(self, test_images: np.ndarray, k_neighbors: int = 50) -> np.ndarray:
        """
        Infer model parameters for multiple test images.
        
        Parameters
        ----------
        test_images : array, shape (n_test, height, width) or (n_test, n_pixels)
        k_neighbors : int
        
        Returns
        -------
        model_params : array, shape (n_test, 2)
            Robust model parameter estimates
        """
        if test_images.ndim == 3:
            n_test = test_images.shape[0]
            test_images = test_images.reshape(n_test, -1)
        
        results = []
        for i, img in enumerate(test_images):
            if (i + 1) % 10 == 0:
                print(f"Processing image {i+1}/{len(test_images)}")
            result = self.infer_model_params(img, k_neighbors=k_neighbors, 
                                            return_uncertainty=False)
            results.append(result['model_params'])
            
        return np.array(results)



if __name__ == "__main__":
    # Test the robust estimator
    print("="*60)
    print("Testing Robust Mean Estimator")
    print("="*60)

    # Configuration
    root_dir = os.getcwd()
    print("Root directory is", root_dir)

    # Global variables that will be set by argument parsing
    USE_PUBLIC_DATASET = True
    MODEL_NAME = 'Robust_Mean_test'
    if not USE_PUBLIC_DATASET:
        DATA_DIR = os.path.join(root_dir, 'input_data/')
    else:
        PUBLIC_DATA_DIR = os.path.join(root_dir, 'input_data/')
        DATA_DIR = PUBLIC_DATA_DIR

    # Load data
    data_obj = Data(data_dir=DATA_DIR, USE_PUBLIC_DATASET=USE_PUBLIC_DATASET)


    load_data(USE_PUBLIC_DATASET)
    config = Config(data_obj.shape, MODEL_NAME)

    # Extract training data (this should be defined in train_HMC.py after load_data)
    # If not available, we need to load it explicitly
    try:
        # Try to use global variables from train_HMC
        X_train, X_val, y_train, y_val
    except NameError:
        # If not available, we need to extract from data_obj
        # This is a simplified extraction - adjust based on actual data structure
        print("Extracting training data from data_obj...")
        # You may need to adjust this based on the actual structure of data_obj
        # For now, using placeholder values
        X_train = data_obj.kappa_train if hasattr(data_obj, 'kappa_train') else np.random.randn(1000, 100)
        y_train = data_obj.label_train if hasattr(data_obj, 'label_train') else np.random.randn(1000, 5)
        X_val = data_obj.kappa_val if hasattr(data_obj, 'kappa_val') else np.random.randn(200, 100)
        y_val = data_obj.label_val if hasattr(data_obj, 'label_val') else np.random.randn(200, 5)

    test_dataset = CosmologyDataset(data_obj.kappa_test, transform=None)

    # Fit robust posterior inference
    print("\nFitting Robust Posterior Inference...")
    model = RobustPosteriorInference(n_components=20, corruption_level=0.15)
    model.fit(X_train, y_train)

        
    # Batch inference
    print("\nBatch inference on 10 test images...")
    predictions = model.infer_batch(X_val[:10], k_neighbors=30)
    errors = np.linalg.norm(predictions - y_val[:10, :2], axis=1)
    print(f"Mean L2 error: {errors.mean():.4f} ± {errors.std():.4f}")
    
    # Test inference on clean test image
    print("\nInferring model parameters for test image...")
    result = model.infer_model_params(test_dataset, k_neighbors=30)
    
    mean_test = result['model_params']
    errorbar_test = result['model_params_std'] if result['model_params_std'] is not None else None

    submission_data = {"means": mean_test.tolist(), "errorbars": errorbar_test.tolist()}
    timestamp = time.strftime("%y-%m-%d-%H-%M")
    zip_filename = f'Submission_{MODEL_NAME}_{timestamp}.zip'
    
    zip_file = Utility.save_json_zip(
        submission_dir="submissions",
        json_file_name="result.json", 
        zip_file_name=zip_filename,
        data=submission_data
    )
    print(f"Test submission saved: {zip_file}")

