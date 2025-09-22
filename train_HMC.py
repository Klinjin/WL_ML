"""
WL ML Uncertainty Challenge - HMC/MCMC Training Script

This script implements two inference methods for cosmological parameter estimation:
1. MCMC: Original Metropolis-Hastings baseline (from CNN_MCMC.ipynb)
2. HMC: Hamiltonian Monte Carlo using numpyro.infer.NUTS

Key improvements over direct methods:
- Better uncertainty quantification via posterior sampling
- Proper handling of parameter correlations
- More robust uncertainty estimates
"""

import os
import json
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import LinearNDInterpolator
from model import BigGANUNet2DModel, ResNetWithAttention, Simple_CNN
import arviz as az

# JAX/Numpyro imports for HMC
try:
    import jax
    import jax.numpy as jnp
    from jax import jit
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import NUTS, MCMC as NumpyroMCMC
    from jax.scipy.stats import multivariate_normal
    from jax.scipy.stats import norm
    HMC_AVAILABLE = True
    print("JAX/Numpyro available - HMC inference enabled")
except ImportError:
    HMC_AVAILABLE = False
    print("JAX/Numpyro not available - only MCMC inference available")

from utils import *
from torchvision import transforms
import argparse

# Configuration
root_dir = os.getcwd()
print("Root directory is", root_dir)

# Global variables that will be set by argument parsing
USE_PUBLIC_DATASET = None
MODEL_NAME = None
DATA_DIR = None
data_obj = None
X_train = None
X_val = None
y_train = None
y_val = None

class Config:
    def __init__(self, data_shape, model_name):
        self.IMG_HEIGHT = data_shape[0]
        self.IMG_WIDTH = data_shape[1]
        
        # Parameters to predict - NOTE: 2 for MCMC approach (not 4 like direct method)
        self.NUM_TARGETS = 2
        
        # Training hyperparameters - matching Simple_CNN baseline
        self.BATCH_SIZE = 64
        self.EPOCHS = 1
        self.LEARNING_RATE = 2e-4
        self.WEIGHT_DECAY = 1e-4
        
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.MODEL_SAVE_PATH = os.path.join(root_dir, f"trained_model/{model_name}.pth")

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc="Training")
    for X, y in pbar:
        X, y = X.to(device), y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, loss_fn, device):
    """Validates the model."""
    model.eval()
    total_loss = 0
    pbar = tqdm(dataloader, total=len(dataloader), desc="Validating")
    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred, y).item()
    
    return total_loss / len(dataloader)


def train_cnn_for_point_estimates(config, USE_PRETRAINED_MODEL=False):
    """Train Simple_CNN for point estimates using MSE loss."""
    
    # Data preprocessing
    means = np.mean(X_train, dtype=np.float32)
    stds = np.std(X_train, dtype=np.float32)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[means], std=[stds]),
    ])
    print(f"Image stats: Mean={means}, Std={stds}")
    
    # Label standardization
    label_scaler = StandardScaler()
    y_train_scaled = label_scaler.fit_transform(y_train)
    y_val_scaled = label_scaler.transform(y_val)
    print(f"Label stats: Mean={label_scaler.mean_}, Std={np.sqrt(label_scaler.var_)}")
    
    # Create datasets
    train_dataset = CosmologyDataset(X_train, y_train_scaled, transform=transform)
    val_dataset = CosmologyDataset(X_val, y_val_scaled, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = Simple_CNN(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_TARGETS).to(config.DEVICE)
    
    if not USE_PRETRAINED_MODEL:
        # Training setup
        loss_fn = nn.MSELoss()  # MSE for point estimates (not KL divergence)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # Training loop
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(config.EPOCHS):
            train_loss = train_epoch(model, train_loader, loss_fn, optimizer, config.DEVICE)
            val_loss = validate_epoch(model, val_loader, loss_fn, config.DEVICE)
            
            scheduler.step(val_loss)
            print(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
                print(f"  -> New best model saved")
        
        end_time = time.time()
        print(f"Training finished in {(end_time - start_time)/60:.2f} minutes.")
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, weights_only=True))
    
    else:
        # Check if the pretrained model exists
        if os.path.exists(config.MODEL_SAVE_PATH):
            print(f"Loading pretrained model from {config.MODEL_SAVE_PATH}")    
            # If the pretrained model exists, load the model
            model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, weights_only=True))

        else:
            # If the pretrained model doesn't exist, show the warning message
            warning_msg = f"The path of pretrained model doesn't exist"
            warnings.warn(warning_msg)
    
    return model, label_scaler, transform


def get_cnn_predictions(model, dataloader, label_scaler, device, nn_error_estimate=False):
    """Get CNN predictions and inverse transform."""
    model.eval()
    predictions = []
    error_estimates = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting CNN predictions"):
            # Handle both cases: with labels (X, y) and without labels (X only)
            if len(batch) == 2:
                X, y = batch  # Has labels
                has_labels = True
            else:
                X = batch     # No labels, just data
                has_labels = False
            
            X = X.to(device)
            pred = model(X)
            pred = label_scaler.inverse_transform(pred.cpu().numpy())
            predictions.append(pred)
            
            # Only compute error estimates if we have labels AND error estimation is requested
            if nn_error_estimate and has_labels:
                y = y.to(device)
                y = label_scaler.inverse_transform(y.cpu().numpy())
                # Compute and store the error estimate
                error = np.abs(pred - y)
                error_estimates.append(error)
            elif nn_error_estimate and not has_labels:
                # If error estimation was requested but no labels available, warn once
                if len(predictions) == 1:  # Only warn on first batch
                    warnings.warn("Error estimation requested but no labels provided in dataloader. Skipping error estimation.")
    
    if nn_error_estimate and len(error_estimates) > 0:
        error_estimates = np.concatenate(error_estimates, axis=0)
        print(f'error_estimates.shape: {error_estimates.shape}')
        mean_error_estimate = error_estimates.mean(axis=0)
        covar_nn = np.cov((error_estimates - mean_error_estimate).T)
        return np.concatenate(predictions, axis=0), mean_error_estimate, covar_nn
    else:
        return np.concatenate(predictions, axis=0)


def setup_mcmc_interpolators(model, val_loader, label_scaler, config):
    """
    Setup mean and covariance interpolators for MCMC likelihood.
    Based on CNN_MCMC.ipynb implementation.
    """
    print("Setting up MCMC interpolators...")
    
    # Get CNN predictions on validation set
    y_pred_val = get_cnn_predictions(model, val_loader, label_scaler, config.DEVICE)
    
    return setup_mcmc_interpolators_from_predictions(y_pred_val, config)


def setup_mcmc_interpolators_from_predictions(y_pred_val, config=None):
    """
    Setup mean and covariance interpolators from existing predictions.
    This avoids redundant prediction calls for efficiency.
    """
    print("Setting up MCMC interpolators from predictions...")
    
    # Get cosmology reference
    cosmology = data_obj.label[:,0,:2]  # Shape: (Ncosmo, 2)
    Ncosmo = len(cosmology)
    
    # Group validation indices by cosmology
    row_to_i = {tuple(cosmology[i]): i for i in range(Ncosmo)}
    index_lists = [[] for _ in range(Ncosmo)]
    
    for idx in range(len(y_val)):
        row_tuple = tuple(y_val[idx])
        i = row_to_i[row_tuple]
        index_lists[i].append(idx)
    
    val_cosmology_idx = [np.array(lst) for lst in index_lists]
    
    # Calculate summary statistics for each cosmology
    n_d = 2  # Number of summary statistics
    d_vector = []
    
    for i in range(Ncosmo):
        d_i = np.zeros((len(val_cosmology_idx[i]), n_d))
        for j, idx in enumerate(val_cosmology_idx[i]):
            d_i[j] = y_pred_val[idx]
        d_vector.append(d_i)
    
    # Calculate mean and covariance
    mean_d_vector = []
    for i in range(Ncosmo):
        mean_d_vector.append(np.mean(d_vector[i], 0))
    mean_d_vector = np.array(mean_d_vector)
    
    # Covariance calculation
    delta = []
    for i in range(Ncosmo):
        delta.append((d_vector[i] - mean_d_vector[i].reshape(1, n_d)))
    
    cov_d_vector = [(delta[i].T @ delta[i] / (len(delta[i])-n_d-2))[None] for i in range(Ncosmo)]
    cov_d_vector = np.concatenate(cov_d_vector, 0)
    
    # Create interpolators
    mean_d_vector_interp = LinearNDInterpolator(cosmology, mean_d_vector, fill_value=np.nan)
    cov_d_vector_interp = LinearNDInterpolator(cosmology, cov_d_vector, fill_value=np.nan)
    
    print(f"Interpolators created for {Ncosmo} cosmologies")
    return mean_d_vector_interp, cov_d_vector_interp, cosmology


def setup_probability_functions(mean_d_vector_interp, cov_d_vector_interp, cosmology=None, mean_error_estimate=None, covar_nn=None):
    """
    Setup shared probability functions for both MCMC and HMC inference.
    Returns both NumPy and JAX versions.
    """
    # Get prior parameters from fitted Gaussians
    (mu_omega_m, sigma_omega_m), (mu_s8, sigma_s8) = Probability.prior(data_obj.label)
    
    print(f"Setting up priors: Ω_m ~ N({mu_omega_m:.3f}, {sigma_omega_m:.3f}²), S_8 ~ N({mu_s8:.3f}, {sigma_s8:.3f}²)")

    
    def log_prior_np(x):
        """
        Log prior for NumPy version - handles both single samples and batches.
        Uses Gaussian priors fitted to training data.
        """
        x = np.atleast_2d(x)  # Ensure 2D array (N_samples, 2)
        
        # Gaussian log prior for each parameter
        log_prior_omega_m = -0.5 * np.log(2 * np.pi * sigma_omega_m**2) - 0.5 * ((x[:, 0] - mu_omega_m) / sigma_omega_m)**2
        log_prior_s8 = -0.5 * np.log(2 * np.pi * sigma_s8**2) - 0.5 * ((x[:, 1] - mu_s8) / sigma_s8)**2
        
        logprior = log_prior_omega_m + log_prior_s8
        
        # Return scalar if input was 1D, array if input was 2D
        return logprior.flatten() if x.shape[0] > 1 or len(x.shape) > 1 else float(logprior[0])
    
    # Gaussian likelihood
    def loglike_np(x, d):
        """
        Log likelihood for NumPy version with proper error handling.
        """
        x = np.atleast_2d(x)  # Ensure 2D array (N_samples, 2)
        d = np.atleast_2d(d)  # Ensure 2D array (N_samples, 2)
        
        try:
            mean = mean_d_vector_interp(x)  # Shape: (N_samples, 2)
            cov = cov_d_vector_interp(x)    # Shape: (N_samples, 2, 2)
            
            if mean is None or cov is None:
                return np.full(x.shape[0], -np.inf)
            
            delta = d - mean  # Shape: (N_samples, 2)
            
            loglike_vals = []
            for i in range(x.shape[0]):
                try:
                    cov_i = cov[i]
                    delta_i = delta[i]
                    
                    # Check for positive definite covariance
                    eigenvals = np.linalg.eigvals(cov_i)
                    if np.any(eigenvals <= 0):
                        loglike_vals.append(-np.inf)
                        continue
                    
                    inv_cov = np.linalg.inv(cov_i)
                    cov_det = np.linalg.slogdet(cov_i)[1]
                    
                    if mean_error_estimate is not None and covar_nn is not None:
                        # Add NN error covariance
                        inv_cov += np.linalg.inv(covar_nn)
                        cov_det += np.linalg.slogdet(covar_nn)[1]
                        delta_i -= mean_error_estimate

                    loglike_i = -0.5 * cov_det - 0.5 * np.dot(delta_i, np.dot(inv_cov, delta_i))
                    loglike_vals.append(loglike_i)
                    
                except (np.linalg.LinAlgError, RuntimeWarning):
                    loglike_vals.append(-np.inf)
            
            return np.array(loglike_vals)
            
        except Exception as e:
            print(f"Warning: likelihood computation failed: {e}")
            return np.full(x.shape[0], -np.inf)
    
    def logp_posterior_np(x, d):
        """
        Log posterior for NumPy version with proper array handling.
        """
        x = np.atleast_2d(x)
        d = np.atleast_2d(d)
        
        logp_prior = log_prior_np(x)
        logp_like = loglike_np(x, d)
        
        logp_posterior = logp_prior + logp_like
        
        # Return appropriate format
        return logp_posterior.flatten() if x.shape[0] > 1 else float(logp_posterior[0])
    
    # JAX versions for HMC
    if HMC_AVAILABLE:

        theta_ranges = Probability.get_theta_ranges(data_obj.label)

        @jit
        def _theta_to_x(theta): #theta is in physical dimension
            x_astro = Probability.bounded_theta_to_x(theta, theta_ranges)
            return jnp.array(x_astro)

        @jit
        def theta_to_x(theta, axis=0): #x is in dimensionless parameter space
            '''
            Transform theta (nsamples, n_params) or (n_params,) to x
            '''
            x_astro = jax.vmap(_theta_to_x, in_axes=axis, out_axes=axis)(jnp.atleast_2d(theta))

            return x_astro.squeeze()
        
        @jit
        def _x_to_theta(x):
            theta_astro = Probability.x_to_bounded_theta(x, theta_ranges)
            return jnp.array(theta_astro)

        @jit
        def x_to_theta(x, axis=0):
            '''
            Transform x (nsamples, n_params) or (n_params,) to theta
            '''
            theta_astro = jax.vmap(_x_to_theta, in_axes=axis, out_axes=axis)(jnp.atleast_2d(x))
            return theta_astro.squeeze()
            
        @jit
        def log_prior_jax(x):
            theta = Probability.x_to_bounded_theta(x, theta_ranges)    
            log_prior_omega_m = norm.logpdf(theta[0], loc=mu_omega_m, scale=sigma_omega_m)
            log_prior_s8 = norm.logpdf(theta[1], loc=mu_s8, scale=sigma_s8)
            log_prior_x = Probability.bounded_variable_ln_dtheta_dx(x)

            return log_prior_omega_m + log_prior_s8 + log_prior_x

        @jit
        def loglike_jax(x, data_x):
            """JAX version of log likelihood using interpolated mean and covariance."""
            # Convert JAX arrays to numpy for interpolation, then back to JAX
            theta = Probability.x_to_bounded_theta(x, theta_ranges)
            data = Probability.x_to_bounded_theta(data_x, theta_ranges)

            # Use the same interpolation as NumPy version
            mean = mean_d_vector_interp(np.array(theta))
            cov = cov_d_vector_interp(np.array(theta))

            if mean_error_estimate is not None and covar_nn is not None:
                data -= jnp.array(mean_error_estimate)
                cov += jnp.array(covar_nn)
            return multivariate_normal.logpdf(data, mean=jnp.array(mean.flatten()), cov=jnp.array(cov[0]))
        
        @jit
        def logp_posterior_jax(x, data_x):
            """JAX version of log posterior."""
            lnP = log_prior_jax(x) + loglike_jax(x, data_x)
            return -lnP
    else:
        log_prior_jax = None
        loglike_jax = None
        logp_posterior_jax = None
        theta_to_x = None
        x_to_theta = None
    
    return {
        'np': {
            'log_prior': log_prior_np,
            'loglike': loglike_np, 
            'logp_posterior': logp_posterior_np
        },
        'jax': {
            'log_prior': log_prior_jax,
            'loglike': loglike_jax,
            'logp_posterior': logp_posterior_jax,
            'theta_to_x': theta_to_x,
            'x_to_theta': x_to_theta
        }
    }


def mcmc_inference(test_predictions, mean_d_vector_interp, cov_d_vector_interp, cosmology, mean_error_estimate=None, covar_nn=None,
                   Nstep=10000, sigma=0.06):
    """
    Original MCMC sampling using Metropolis-Hastings.
    Based exactly on CNN_MCMC.ipynb implementation.
    """
    print("Running MCMC inference...")
    
    # Setup probability functions
    prob_funcs = setup_probability_functions(mean_d_vector_interp, cov_d_vector_interp, cosmology, mean_error_estimate, covar_nn)
    logp_posterior = prob_funcs['np']['logp_posterior']
    
    # MCMC sampling
    Ntest = len(test_predictions)
    current = cosmology[np.random.choice(len(cosmology), size=Ntest)] # Shape: (Ntest, 2)
    
    # Debug: Test initial probability evaluation
    print(f"Test predictions shape: {test_predictions.shape}")
    print(f"Current initial shape: {current.shape}")
    print(f"Sample initial position: {current[0]} for test: {test_predictions[0]}")
    
    curr_logprob = logp_posterior(current, test_predictions)
    print(f"Initial log probabilities range: [{np.min(curr_logprob):.2f}, {np.max(curr_logprob):.2f}]")
    
    # Check for valid initial probabilities
    valid_initial = np.isfinite(curr_logprob)
    print(f"Valid initial probabilities: {np.sum(valid_initial)}/{len(curr_logprob)}")
    
    if np.sum(valid_initial) == 0:
        print("ERROR: No valid initial probabilities! Check interpolation setup.")
        return np.zeros((Ntest, 2)), np.ones((Ntest, 2)) * 0.1  # Fallback
    
    states = []
    total_acc = np.zeros(len(current))
    
    print(f"Running {Nstep} MCMC steps for {Ntest} test samples...")
    start_time = time.time()
    
    for i in tqdm(range(Nstep), desc="MCMC sampling"):
        # Generate proposals
        proposal = current + np.random.randn(*current.shape) * sigma
        proposal_logprob = logp_posterior(proposal, test_predictions)
        
        # Acceptance probability
        acc_logprob = proposal_logprob - curr_logprob
        acc_logprob[acc_logprob > 0] = 0
        acc_prob = np.exp(acc_logprob)
        
        # Accept/reject
        acc = np.random.uniform(size=len(current)) < acc_prob
        total_acc += acc_prob
        
        # Update states
        current[acc] = proposal[acc]
        curr_logprob[acc] = proposal_logprob[acc]
        
        states.append(np.copy(current)[None])
        
        # Progress update
        if i % (Nstep // 10) == (Nstep // 10) - 1:
            elapsed = time.time() - start_time
            acceptance_rate = np.mean(total_acc / (i + 1))
            print(f"  Step {i+1}/{Nstep}, Time: {elapsed:.1f}s, Acceptance: {acceptance_rate:.3f}")
            start_time = time.time()
    
    # Remove burn-in and compute statistics
    states = np.concatenate(states[int(0.2*Nstep):], 0)  # Remove first 20%
    mean_posterior = np.mean(states, 0)
    std_posterior = np.std(states, 0)
    
    print(f"MCMC completed. Final acceptance rate: {np.mean(total_acc / Nstep):.3f}")
    
    return mean_posterior, std_posterior


def hmc_inference(test_predictions, mean_d_vector_interp, cov_d_vector_interp, cosmology, mean_error_estimate=None, covar_nn=None,
                  num_samples=8000, num_warmup=2000, num_chains=4, max_tree_depth=10):
    """
    HMC sampling using numpyro.infer.NUTS with proper model definition.
    """
    if not HMC_AVAILABLE:
        raise ImportError("JAX/Numpyro not available for HMC inference")
    
    print("Running HMC inference with NUTS...")
    
    # Setup shared probability functions
    prob_funcs = setup_probability_functions(mean_d_vector_interp, cov_d_vector_interp, cosmology, mean_error_estimate, covar_nn)
    
    results_list = []
    
    print(f"Running HMC for {len(test_predictions)} test samples...")
    
    # Get prior parameters from fitted Gaussians
    (mu_omega_m, sigma_omega_m), (mu_s8, sigma_s8) = Probability.prior(data_obj.label)
    
    def model(test_pred):
        """Numpyro model definition for HMC."""
        # Define priors
        omega_m = numpyro.sample("omega_m", dist.Normal(mu_omega_m, sigma_omega_m))
        s8 = numpyro.sample("s8", dist.Normal(mu_s8, sigma_s8))
        
        theta = jnp.array([omega_m, s8])
        
        # Get interpolated mean and covariance
        theta_np = np.array(theta)
        mean = mean_d_vector_interp(theta_np.reshape(1, -1))
        cov = cov_d_vector_interp(theta_np.reshape(1, -1))
        
        # Add NN error if provided
        if mean_error_estimate is not None and covar_nn is not None:
            test_pred_corrected = test_pred - jnp.array(mean_error_estimate)
            cov_total = jnp.array(cov) + jnp.array(covar_nn)
        else:
            test_pred_corrected = test_pred
            cov_total = jnp.array(cov)
        
        # Likelihood
        numpyro.sample("obs", dist.MultivariateNormal(jnp.array(mean.flatten()), cov_total), obs=test_pred_corrected)
    
    for i, test_pred in enumerate(tqdm(test_predictions, desc="HMC sampling")):
        test_pred_jax = jnp.array(test_pred)
        
        # Setup NUTS sampler
        nuts_kernel = NUTS(
            model,
            adapt_step_size=True, 
            dense_mass=True, 
            max_tree_depth=max_tree_depth
        )
        
        # Setup MCMC with vectorized chains
        mcmc = NumpyroMCMC(
            nuts_kernel, 
            num_warmup=num_warmup, 
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method='vectorized'
        )
        
        # Run MCMC for this test sample
        rng_key = jax.random.PRNGKey(i)
        
        try:
            start_time = time.time()
            mcmc.run(rng_key, test_pred_jax)
            total_time = time.time() - start_time
            print(f"HMC sampling completed in {total_time:.2f} seconds")

            # Extract samples
            samples = mcmc.get_samples()
            omega_m_samples = samples['omega_m']
            s8_samples = samples['s8']
            
            # Compute posterior statistics
            mean_omega_m = jnp.mean(omega_m_samples)
            mean_s8 = jnp.mean(s8_samples)
            std_omega_m = jnp.std(omega_m_samples)
            std_s8 = jnp.std(s8_samples)
        

            # Compute the neff and summarize cost
            az_summary = az.summary(az.from_numpyro(mcmc))
            neff = az_summary["ess_bulk"].to_numpy()
            neff_mean = np.mean(neff)
            r_hat = az_summary["r_hat"].to_numpy()
            r_hat_mean = np.mean(r_hat)
            sec_per_neff = (total_time / neff_mean)
            ms_per_neff = 1e3 * sec_per_neff

            results_list.append({
                'mean': [float(mean_omega_m), float(mean_s8)],
                'std': [float(std_omega_m), float(std_s8)],
                'neff_mean': float(neff_mean),
                'ms_per_neff': float(ms_per_neff),
                'r_hat_mean': float(r_hat_mean)
            })
            
        except Exception as e:
            print(f"HMC failed for sample {i}: {e}")
        
        # Progress update
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{len(test_predictions)} samples")
    
    # Convert to arrays
    means = np.array([r['mean'] for r in results_list])
    stds = np.array([r['std'] for r in results_list])

    neffs_mean = np.array([r['neff_mean'] for r in results_list]).mean()    
    ms_per_neffs_mean = np.array([r['ms_per_neff'] for r in results_list]).mean()
    r_hats_mean = np.array([r['r_hat_mean'] for r in results_list]).mean()

    print(f"HMC inference completed for {len(test_predictions)} samples with {num_chains} chains, {num_samples} samples each")
    print(f"Average neff: {neffs_mean:.1f}")
    print(f"Average r_hat: {r_hats_mean:.3f})")
    print(f"Average ms/neff: {ms_per_neffs_mean:.1f} ms")

    return means, stds

def hmc_inference_x_transform(test_predictions, mean_d_vector_interp, cov_d_vector_interp, cosmology, mean_error_estimate=None, covar_nn=None,
                  num_samples=8000, num_warmup=2000, num_chains=4, max_tree_depth=10):
    """
    HMC sampling using numpyro.infer.NUTS with proper model definition.
    """
    if not HMC_AVAILABLE:
        raise ImportError("JAX/Numpyro not available for HMC inference")
    
    print("Running HMC inference with NUTS...")
    
    # Setup shared probability functions
    prob_funcs = setup_probability_functions(mean_d_vector_interp, cov_d_vector_interp, cosmology, mean_error_estimate, covar_nn)
    logp_posterior = prob_funcs['jax']['logp_posterior']
    theta_to_x = prob_funcs['jax']['theta_to_x']
    x_to_theta = prob_funcs['jax']['x_to_theta']

    @jit
    def numpyro_potential_fun(data_x): #potential function for numpyro
        return jax.tree_util.Partial(logp_posterior, data_x=data_x)

    results_list = []
    
    print(f"Running HMC for {len(test_predictions)} test samples...")
    test_predictions_x = theta_to_x(jnp.array(test_predictions))

    for i, test_pred_x in enumerate(tqdm(test_predictions_x, desc="HMC sampling")):

        # Setup NUTS sampler
        nuts_kernel = NUTS(
            potential_fn=numpyro_potential_fun(test_pred_x),
            adapt_step_size=True, 
            dense_mass=True, 
            max_tree_depth=max_tree_depth
        )
        
        # Setup MCMC with vectorized chains
        mcmc = NumpyroMCMC(
            nuts_kernel, 
            num_warmup=num_warmup, 
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method='vectorized'
        )
        
        # Run MCMC for this test sample
        rng_key = jax.random.PRNGKey(i)
        
        try:
            start_time = time.time()
            mcmc.run(rng_key, test_pred_x)
            total_time = time.time() - start_time
            print(f"HMC sampling completed in {total_time:.2f} seconds")

            # Extract samples
            x_samples = mcmc.get_samples(group_by_chain=False)
            samples = x_to_theta(x_samples)
            omega_m_samples = samples[:,0]
            s8_samples = samples[:,1]

            # Compute posterior statistics
            mean_omega_m = jnp.mean(omega_m_samples)
            mean_s8 = jnp.mean(s8_samples)
            std_omega_m = jnp.std(omega_m_samples)
            std_s8 = jnp.std(s8_samples)

            # Compute the neff and summarize cost
            az_summary = az.summary(az.from_numpyro(mcmc))
            neff = az_summary["ess_bulk"].to_numpy()
            neff_mean = np.mean(neff)
            r_hat = az_summary["r_hat"].to_numpy()
            r_hat_mean = np.mean(r_hat)
            sec_per_neff = (total_time / neff_mean)
            ms_per_neff = 1e3 * sec_per_neff

            results_list.append({
                'mean': [float(mean_omega_m), float(mean_s8)],
                'std': [float(std_omega_m), float(std_s8)],
                'neff_mean': float(neff_mean),
                'ms_per_neff': float(ms_per_neff),
                'r_hat_mean': float(r_hat_mean)
            })
            
        except Exception as e:
            print(f"HMC failed for sample {i}: {e}")
        
        # Progress update
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{len(test_predictions)} samples")
    
    # Convert to arrays
    means = np.array([r['mean'] for r in results_list])
    stds = np.array([r['std'] for r in results_list])

    neffs_mean = np.array([r['neff_mean'] for r in results_list]).mean()    
    ms_per_neffs_mean = np.array([r['ms_per_neff'] for r in results_list]).mean()
    r_hats_mean = np.array([r['r_hat_mean'] for r in results_list]).mean()

    print(f"HMC inference completed for {len(test_predictions)} samples with {num_chains} chains, {num_samples} samples each")
    print(f"Average neff: {neffs_mean:.1f}")
    print(f"Average r_hat: {r_hats_mean:.3f})")
    print(f"Average ms/neff: {ms_per_neffs_mean:.1f} ms")

    return means, stds

def load_data(use_public_dataset):
    """Load and prepare the training data."""
    global data_obj, X_train, X_val, y_train, y_val, DATA_DIR
    
    # Data setup
    if not use_public_dataset:
        DATA_DIR = os.path.join(root_dir, 'input_data/')
    else:
        PUBLIC_DATA_DIR = os.path.join(root_dir, 'input_data/')
        DATA_DIR = PUBLIC_DATA_DIR

    # Load data
    data_obj = Data(data_dir=DATA_DIR, USE_PUBLIC_DATASET=use_public_dataset)
    data_obj.load_test_data()
    Ncosmo = data_obj.Ncosmo
    Nsys = data_obj.Nsys
    ng = data_obj.ng

    print(f'There are {Ncosmo} cosmological models, each has {Nsys} realizations of nuisance parameters in the training data.')
    print(f'We assume a galaxy number density of {ng} per arcmin², which determines the noise level of the experiment.')

    # Load training data
    noisy_kappa_train = Utility.load_np(data_dir=DATA_DIR, file_name="noisy_kappa_train.npy")
    label_train = Utility.load_np(data_dir=DATA_DIR, file_name="label_train.npy")
    noisy_kappa_val = Utility.load_np(data_dir=DATA_DIR, file_name="noisy_kappa_val.npy")
    label_val = Utility.load_np(data_dir=DATA_DIR, file_name="label_val.npy")

    Ntrain = label_train.shape[0]*label_train.shape[1]
    Nval = label_val.shape[0]*label_val.shape[1]
    print('Training and validation data loaded')

    # Reshape data for CNN
    X_train = noisy_kappa_train.reshape(Ntrain, *data_obj.shape)
    X_val = noisy_kappa_val.reshape(Nval, *data_obj.shape)

    # Only keep cosmological parameters (Omega_m, S_8)
    y_train = label_train.reshape(Ntrain, 5)[:, :2]
    y_val = label_val.reshape(Nval, 5)[:, :2]

    print(f'Shape of training data = {X_train.shape}')
    print(f'Shape of validation data = {X_val.shape}')
    print(f'Shape of training labels = {y_train.shape}')
    print(f'Shape of validation labels = {y_val.shape}')


def main():
    """Main training and inference pipeline."""
    def create_argparser():
        parser = argparse.ArgumentParser(description='WL ML Uncertainty Challenge - HMC/MCMC Training Script')
        parser.add_argument('--method', choices=['mcmc', 'hmc'], default='mcmc',
                           help='Inference method to use (default: mcmc)')
        parser.add_argument('--use-public-dataset', action='store_true', default=True,
                           help='Use public dataset (default: True)')
        parser.add_argument('--model-name', type=str, default='Simple_CNN_HMC_baseline',
                           help='Name for the model and output files (default: Simple_CNN_HMC_baseline)')
        parser.add_argument('--nn-error-estimate', action='store_true', default=False,  
                           help='Estimate NN error from validation set (default: False)')
        parser.add_argument('--pretrained', action='store_true', default=False,
                           help='Use pretrained model (default: False)')
        return parser
    
    args = create_argparser().parse_args()
    
    # Set global variables and inference method
    global USE_PUBLIC_DATASET, MODEL_NAME
    USE_PUBLIC_DATASET = args.use_public_dataset
    MODEL_NAME = args.model_name
    NN_ERROR_ESTIMATE = args.nn_error_estimate
    USE_PRETRAINED_MODEL = args.pretrained
    inference_method = args.method
    
    # Set method suffix for file naming
    method_suffix = "HMC" if (inference_method == 'hmc' and HMC_AVAILABLE) else "MCMC"
    if inference_method == 'hmc' and not HMC_AVAILABLE:
        print("HMC not available, falling back to MCMC...")
    
    print(f"Configuration: USE_PUBLIC_DATASET={USE_PUBLIC_DATASET}, MODEL_NAME={MODEL_NAME}, METHOD={method_suffix}")
    
    # Load data
    load_data(USE_PUBLIC_DATASET)
    
    # Create config with loaded data
    config = Config(data_obj.shape, MODEL_NAME)
    print(f"Using device: {config.DEVICE}")
    
    # Step 1: Train CNN for point estimates/ Load Pretrained model
    print("\n=== Step 1: Training CNN for Point Estimates ===")
    model, label_scaler, transform = train_cnn_for_point_estimates(config, USE_PRETRAINED_MODEL=USE_PRETRAINED_MODEL)

    # Step 2: Get all predictions (validation and test) - single call for efficiency
    print("\n=== Step 2: Getting CNN Predictions ===")
    
    # Create validation dataset - include labels if error estimation is needed
    if NN_ERROR_ESTIMATE:
        # Need labels for error estimation - use scaled labels as in training
        y_val_scaled = label_scaler.transform(y_val)
        val_dataset = CosmologyDataset(X_val, y_val_scaled, transform=transform)
    else:
        # No error estimation needed - just use input data
        val_dataset = CosmologyDataset(X_val, transform=transform)
        
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    test_dataset = CosmologyDataset(data_obj.kappa_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Get predictions with optional error estimation
    if NN_ERROR_ESTIMATE:   
        print("Estimating NN error from validation set...")
        val_predictions, mean_error_estimate, covar_nn = get_cnn_predictions(
            model, val_loader, label_scaler, config.DEVICE, NN_ERROR_ESTIMATE)
    else:
        val_predictions = get_cnn_predictions(model, val_loader, label_scaler, config.DEVICE, NN_ERROR_ESTIMATE)
        mean_error_estimate = None
        covar_nn = None
    
    test_predictions = get_cnn_predictions(model, test_loader, label_scaler, config.DEVICE, False)
    print(f"Validation predictions shape: {val_predictions.shape}")
    print(f"Test predictions shape: {test_predictions.shape}")
    
    # Step 3: Setup MCMC interpolators using validation predictions
    print("\n=== Step 3: Setting up MCMC Interpolators ===")
    mean_d_vector_interp, cov_d_vector_interp, cosmology = setup_mcmc_interpolators_from_predictions(
        val_predictions, config)
    
    # Step 4: Validation scoring
    print("\n=== Step 4: Validation Scoring ===")
    if inference_method == 'hmc' and HMC_AVAILABLE:
        print("Using HMC inference for validation...")
        mean_val, errorbar_val = hmc_inference(
            val_predictions, mean_d_vector_interp, cov_d_vector_interp, cosmology, mean_error_estimate, covar_nn)
    else:
        print("Using MCMC inference for validation...")
        mean_val, errorbar_val = mcmc_inference(
            val_predictions, mean_d_vector_interp, cov_d_vector_interp, cosmology, mean_error_estimate, covar_nn)
    
    # Calculate validation score with error handling
    try:
        validation_score = Score._score_phase1(y_val, mean_val, errorbar_val)
        print(f'Validation score: {validation_score:.6f}')
        print(f'Average error bar: {np.mean(errorbar_val, 0)}')
        
        # Additional validation metrics
        mse_omega_m = np.mean((y_val[:, 0] - mean_val[:, 0])**2)
        mse_s8 = np.mean((y_val[:, 1] - mean_val[:, 1])**2)
        print(f'MSE: Ω_m={mse_omega_m:.6f}, S_8={mse_s8:.6f}')
        
    except Exception as e:
        print(f"Error calculating validation score: {e}")
        validation_score = -999.0
    
    # Step 5: Test inference
    print("\n=== Step 5: Test Inference ===")
    if inference_method == 'hmc' and HMC_AVAILABLE:
        print("Using HMC inference for test predictions...")
        mean_test, errorbar_test = hmc_inference(
            test_predictions, mean_d_vector_interp, cov_d_vector_interp, cosmology, mean_error_estimate, covar_nn)
    else:
        print("Using MCMC inference for test predictions...")
        mean_test, errorbar_test = mcmc_inference(
            test_predictions, mean_d_vector_interp, cov_d_vector_interp, cosmology, mean_error_estimate, covar_nn)
    
    # Step 6: Save results
    print("\n=== Step 6: Saving Results ===")
    
    # Save validation results
    scoring_output = {
        "validation_score": float(validation_score),
        "method": method_suffix,
        "num_val": len(y_val),
        "num_train": len(y_train),
        "model": "Simple_CNN"
    }
    
    # Ensure output directory exists
    os.makedirs("scoring_output", exist_ok=True)
    
    output_filename = f"scoring_output/{args.model_name}_{method_suffix}.json"
    with open(output_filename, "w") as f:
        json.dump(scoring_output, f, indent=2)
    print(f"Validation results saved to {output_filename}")
    
    # Save test submission
    submission_data = {"means": mean_test.tolist(), "errorbars": errorbar_test.tolist()}
    timestamp = time.strftime("%y-%m-%d-%H-%M")
    zip_filename = f'Submission_{args.model_name}_{method_suffix}_{timestamp}.zip'
    
    zip_file = Utility.save_json_zip(
        submission_dir="submissions",
        json_file_name="result.json", 
        zip_file_name=zip_filename,
        data=submission_data
    )
    print(f"Test submission saved: {zip_file}")
    
    return validation_score


if __name__ == "__main__":
    validation_score = main()
    print(f"\nFinal validation score: {validation_score:.6f}")