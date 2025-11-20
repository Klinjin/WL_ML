import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm


class SEVERFilter:
    """
    SEVER: Outlier detection via gradient analysis
    
    For weak lensing: Detects maps with anomalous gradients
    that distort cosmological parameter estimates
    """

    def __init__(self, model, variance_threshold=1.5, sigma=None, top_fraction=None, num_target=4):
        """
        Args:
            model: Trained CNN (Simple_CNN)
            variance_threshold: Filter threshold (c in Algorithm 2)
            sigma: Approximate learner regularization
        """
        self.model = model
        self.variance_threshold = variance_threshold
        self.sigma = sigma
        self.top_fraction = top_fraction  # Fraction of samples to remove
        self.num_target = num_target

        
    def compute_gradients(self, dataloader, loss_fn, device):
        """
        Compute gradients for all samples in dataset
        Returns: gradient_matrix [n_samples x n_params]
        """
        # Set model to eval mode but enable gradients
        self.model.eval()
        all_gradients = []
        
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X.requires_grad = True
            
            # Forward pass
            if self.num_target == 4:
                pred_means, pred_sigmas = self.model(X)
                loss = loss_fn(pred_means, pred_sigmas, y)
            else:
                pred_means = self.model(X)
                loss = loss_fn(pred_means, y)
            # Compute gradient w.r.t. predictions (not weights!)
            grad = torch.autograd.grad(loss, pred_means, 
                                      retain_graph=True)[0]
            all_gradients.append(grad.detach().cpu().numpy())
        
        return np.concatenate(all_gradients, axis=0)
    
    def detect_outliers(self, gradient_matrix):
        """
        SEVER Algorithm 1 & 2: Detect outliers via SVD
        
        Returns:
            inlier_mask: Boolean mask for clean samples
            outlier_scores: Score for each sample
        """
        # Center gradients (Original: normalzed by sqrt(n_sample))
        G_centered = gradient_matrix - np.mean(gradient_matrix, axis=0)
        
        # SVD
        U, S, Vt = np.linalg.svd(G_centered, full_matrices=False)
        v = Vt[0]
        
        # Outlier scores
        outlier_scores = ((G_centered @ v) ** 2)
        
        # Using average variance (normalized by n)
        avg_variance = np.mean(outlier_scores)
        
        # Estimate sigma from the data itself (more robust)
        if self.sigma is None:
            sigma_estimate = np.sqrt(np.mean(outlier_scores))
        else:
            sigma_estimate = self.sigma
        threshold = self.variance_threshold * sigma_estimate 
        
        if avg_variance <= threshold:
            print(f"  Clean data detected (variance={avg_variance:.2f} ≤ {threshold:.2f})")
            return np.ones(len(outlier_scores), dtype=bool), outlier_scores
        
        if self.top_fraction is None:
            max_score = np.max(outlier_scores) if outlier_scores.size > 0 else 0.0
            T = np.random.uniform(0, max_score)
            print(f"Remove with random threshold {T}")
            # Algorithm 2, line 4: Keep points where τᵢ < T
            inlier_mask = outlier_scores < T
            
            n_remove = np.sum(~inlier_mask)
        elif np.quantile(outlier_scores, 1-self.top_fraction) > 0:
            # Practical version: Remove top-p% deterministically
            print(f"Remove {self.top_fraction} fraction of scores")
            outlier_scores = outlier_scores / np.quantile(outlier_scores, 1-self.top_fraction)
            inlier_mask = outlier_scores < 1.0
            n_remove = np.sum(~inlier_mask)
        else:
            print(f"Remove {self.top_fraction} fraction of scores would remove everything --> remove nothing")
            return np.ones(len(outlier_scores), dtype=bool), outlier_scores/np.max(outlier_scores)

        print(f"  Removing {n_remove} points (top {n_remove/len(outlier_scores)*100:.2f}%)")
        print(f"  Variance: {avg_variance:.2e} > threshold {threshold:.2e}")
        
        return inlier_mask, outlier_scores
    
    def iterative_filtering(self, dataloader, loss_fn, device):

        original_indices = np.arange(len(dataloader.dataset))
        original_size = len(original_indices)        

        # Compute gradients on current subset
        gradients = self.compute_gradients(dataloader, loss_fn, device)
            
        # Detect outliers with SVD
        inlier_mask, scores = self.detect_outliers(gradients)
            
        current_indices = original_indices[inlier_mask]

        n_removed = len(original_indices)-len(current_indices)
    
        # Report final statistics
        if n_removed > 0:
            print(f"  Final dataset: {len(current_indices)} samples ({100*len(current_indices)/original_size:.1f}% retained)")
        
        return current_indices, scores

class TestSetSEVER:
    """
    SEVER adaptation for test-time outlier detection
    Uses prediction patterns instead of gradients
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def compute_prediction_features(self, dataloader):
        """
        Extract features for outlier detection:
        - Predicted means (Omega_m, S_8)
        - Predicted uncertainties (sigma_Omega_m, sigma_S_8)
        - Intermediate CNN features
        """
        self.model.eval()
        all_means, all_sigmas, all_features = [], [], []
        
        with torch.no_grad():
            for X in dataloader:
                X = X.to(self.device)
                
                # Get predictions
                means, sigmas = self.model(X)
                
                # Extract intermediate features (before final FC layer)
                features = self.model.conv_stack(X)
                features = torch.flatten(features, start_dim=1)
                
                all_means.append(means.cpu().numpy())
                all_sigmas.append(sigmas.cpu().numpy())
                all_features.append(features.cpu().numpy())
        
        means = np.concatenate(all_means, axis=0)
        sigmas = np.concatenate(all_sigmas, axis=0)
        features = np.concatenate(all_features, axis=0)
        
        return means, sigmas, features
    
    def detect_outliers_unsupervised(self, means, sigmas, features,
                                    threshold_percentile=95):
        """
        Unsupervised outlier detection on test set
        
        Strategy:
        1. Identify samples with extreme uncertainties
        2. Detect anomalies in feature space via SVD
        3. Combine signals for robust filtering
        """
        # Signal 1: Extreme uncertainties
        uncertainty_scores = np.sum(sigmas ** 2, axis=1)
        uncertainty_threshold = np.percentile(uncertainty_scores, 
                                             threshold_percentile)
        
        # Signal 2: Feature space anomalies (via PCA/SVD)
        features_centered = features - np.mean(features, axis=0)
        U, S, Vt = np.linalg.svd(features_centered, full_matrices=False)
        
        # Project onto top PC (captures main variation)
        # Outliers deviate from main subspace
        reconstruction = (features_centered @ Vt[:10].T) @ Vt[:10]
        reconstruction_error = np.sum((features_centered - reconstruction)**2, 
                                     axis=1)
        
        feature_threshold = np.percentile(reconstruction_error,
                                         threshold_percentile)
        
        # Combine signals
        is_outlier = ((uncertainty_scores > uncertainty_threshold) | 
                     (reconstruction_error > feature_threshold))
        
        return ~is_outlier, uncertainty_scores, reconstruction_error
    

def robust_test_prediction(model, dataloader, config, label_scaler, n_monte_carlo=10):
    """
    Make robust predictions using Monte Carlo dropout
    
    Args:
        model: Trained neural network
        dataloader: DataLoader for validation/test data
        config: Configuration object
        label_scaler: StandardScaler for inverse transforming predictions
        n_monte_carlo: Number of MC samples for uncertainty estimation
    
    Returns:
        mean_predictions: (N, 2) array of predicted cosmological parameters
        std_predictions: (N, 2) array of uncertainty estimates
    """
    
    # Enable dropout during inference for MC sampling
    def enable_dropout(model):
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    model.eval()
    
    all_predictions = []
    all_uncertainties = []
    
    with torch.no_grad():
        for X, _ in tqdm(dataloader, desc="MC Sampling"):
            X = X.to(config.DEVICE)
            
            # Collect predictions from multiple forward passes
            batch_preds = []
            batch_sigmas = []
            
            for _ in range(n_monte_carlo):
                # Enable dropout for MC sampling
                enable_dropout(model)
                
                pred_means, pred_sigmas = model(X)
                batch_preds.append(pred_means.cpu().numpy())
                batch_sigmas.append(pred_sigmas.cpu().numpy())
            
            # Average predictions across MC samples
            batch_preds = np.array(batch_preds)  # (n_monte_carlo, batch_size, 2)
            batch_sigmas = np.array(batch_sigmas)
            
            # Mean prediction
            mean_pred = np.mean(batch_preds, axis=0)  # (batch_size, 2)
            
            # Combined uncertainty (epistemic + aleatoric)
            epistemic_uncertainty = np.std(batch_preds, axis=0)  # Prediction variance
            aleatoric_uncertainty = np.mean(batch_sigmas, axis=0)  # Model's predicted sigma
            
            # Total uncertainty
            total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
            
            all_predictions.append(mean_pred)
            all_uncertainties.append(total_uncertainty)
    
    # Concatenate all batches
    mean_predictions = np.concatenate(all_predictions, axis=0)
    std_predictions = np.concatenate(all_uncertainties, axis=0)
    
    # Inverse transform to original scale
    mean_predictions = label_scaler.inverse_transform(mean_predictions)
    std_predictions = std_predictions * np.sqrt(label_scaler.var_)
    
    return mean_predictions, std_predictions

