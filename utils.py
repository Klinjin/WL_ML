import os
import json
import time
import zipfile
import datetime
import warnings

from triton import jit
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.stats import norm


root_dir = os.getcwd()
print("Root directory is", root_dir)

class Probability:
        
    @staticmethod 
    def prior(label): #data_obj.viz_label

        omega_m_values = label[:, 0, 0]  # Shape: (N_cosmo,)
        s8_values = label[:, 0, 1]       # Shape: (N_cosmo,)

        params = norm.fit(omega_m_values)
        mu_omega_m, sigma_omega_m = params[0], params[1]

        params = norm.fit(s8_values)
        mu_s8, sigma_s8 = params[0], params[1]

        return (mu_omega_m, sigma_omega_m), (mu_s8, sigma_s8)

    @staticmethod
    def get_theta_ranges(label):
        """
        Get the ranges of the cosmological parameters from the label data.

        Args:
            label (np.ndarray): Array of shape (N_cosmo, N_sys, N_params) containing the parameter values.

        Returns:
            theta_ranges (list): List of tuples specifying the (min, max) range for each parameter.
        """
        theta_ranges = []
        for i in range(label.shape[2]):
            param_values = label[:, :, i].flatten()
            theta_ranges.append((np.min(param_values), np.max(param_values)))
        return theta_ranges

    @jit
    def bounded_variable_ln_dtheta_dx(x, ln_p_theta, theta_ranges):
        """
        For a parameter distributed with prior P(theta) on the range (theta_0, theta_1), we instead
        transform to the unbounded parameter x, via:

            theta  = theta_0 + (theta_1 - theta_0)*sigmoid(x)

        The prior on theta is given as input ln_p_theta.

        The Jacobian Prob(x) = P(theta) dtheta/dx then implies

            Prob(x) = P(theta) * sigmoid(x) * (1 - sigmoid(x)) * (theta_1 - theta_0)

        which implies the lnProb(x) given by this function.

        Args:
            x (float or jax.numpy.ndarray):
                Unbounded parameter vector.
            ln_p_theta (float or jax.numpy.ndarray):
                Log prior probability P(theta) at the transformed parameter theta.
            theta_ranges (tuple or list):
                Range (theta_0, theta_1) for the bounded parameter.

        Returns:
            lnP (jax.numpy.ndarray):
                Log prior probability at the parameter vector. The output shape/type is the same as the input.
        """
        
        # Calculate the range width
        theta_range_width = theta_ranges[1] - theta_ranges[0]
        
        # Log of the Jacobian transformation
        log_jacobian = jax.nn.log_sigmoid(x) + jnp.log(1.0 - jax.nn.sigmoid(x)) + jnp.log(theta_range_width)
        
        return ln_p_theta + log_jacobian
    
    @jit
    def bounded_theta_to_x(theta, theta_ranges):
        """
        Transform a bounded paramter vector theta into an unbounded parameter vector x using a logit transformation.
        This is the single vector function called by the vectorized function in base.

        Args:
            theta (jax.numpy.ndarray):
                Parameter vector with shape=(n_params,)
            theta_range (list):
                List of length n_params containing 2-d tuples, where each tuple is the range of the parameter.
                The first element of the tuple is the lower bound, and the second element is the upper bound.

        Returns:
            x (jax.numpy.ndarray):
                Transformed parameter vector with shape=(n_params,)

        """

        _theta = jnp.atleast_1d(theta)
        n_params = _theta.shape[0]
        x = jnp.zeros(n_params)

        for i in range(n_params):
            x = x.at[i].set(Probability._bounded_theta_to_x(_theta[i], theta_ranges[i]))

        return jnp.array(x)

    @jit
    def _bounded_theta_to_x(theta_element, theta_range):
        """
        Transform an element of a bounded parameter vector theta into an element of an unbounded parameter vector x using a
        logit transformation. This is the single element function called by the functions above.

        Args:
            theta_element (float):
                Element of a parameter vector.
            theta_range (tuple):
                A tuple of length=2 specifying the range of the parameter.
                The first element of the tuple is the lower bound, and the second element is the upper bound.

        Returns:
            x_element (float):
                Transformed parameter vector element.

        """

        return jax.scipy.special.logit(
            jnp.clip((theta_element - theta_range[0])/(theta_range[1] - theta_range[0]), a_min=1e-7, a_max=1.0 - 1e-7))


class Utility:
    @staticmethod
    def add_noise(data, mask, ng, pixel_size=2.):
        """
        Add noise to a noiseless convergence map.

        Parameters
        ----------
        data : np.array
            Noiseless convergence maps.
        mask : np.array
            Binary mask map.
        ng : float
            Number of galaxies per arcmin². This determines the noise level; a larger number means smaller noise.
        pixel_size : float, optional
            Pixel size in arcminutes (default is 2.0).
        """

        return data + np.random.randn(*data.shape) * 0.4 / (2*ng*pixel_size**2)**0.5 * mask

    @staticmethod
    def load_np(data_dir, file_name):
        file_path = os.path.join(data_dir, file_name)
        return np.load(file_path)

    @staticmethod
    def save_np(data_dir, file_name, data):
        file_path = os.path.join(data_dir, file_name)
        np.save(file_path, data)

    @staticmethod
    def save_json_zip(submission_dir, json_file_name, zip_file_name, data):
        """
        Save a dictionary with 'means' and 'errorbars' into a JSON file,
        then compress it into a ZIP file inside submission_dir.

        Parameters
        ----------
        submission_dir : str
            Path to the directory where the ZIP file will be saved.
        file_name : str
            Name of the ZIP file (without extension).
        data : dict
            Dictionary with keys 'means' and 'errorbars'.

        Returns
        -------
        str
            Path to the created ZIP file.
        """
        os.makedirs(submission_dir, exist_ok=True)

        json_path = os.path.join(submission_dir, json_file_name)

        # Save JSON file
        with open(json_path, "w") as f:
            json.dump(data, f)

        # Path to ZIP
        zip_path = os.path.join(submission_dir, zip_file_name)

        # Create ZIP containing only the JSON
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(json_path, arcname=json_file_name)

        # Remove the standalone JSON after zipping
        os.remove(json_path)

        return zip_path

class Data:
    def __init__(self, data_dir, USE_PUBLIC_DATASET):
        self.USE_PUBLIC_DATASET = USE_PUBLIC_DATASET
        self.data_dir = data_dir
        self.mask_file = 'WIDE12H_bin2_2arcmin_mask.npy'
        self.viz_label_file = 'label.npy'

        if self.USE_PUBLIC_DATASET:
            self.kappa_file = 'WIDE12H_bin2_2arcmin_kappa.npy'
            self.label_file = self.viz_label_file
            self.Ncosmo = 101  # Number of cosmologies in the entire training data
            self.Nsys = 256    # Number of systematic realizations in the entire training data
            self.test_kappa_file = 'WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'
            self.Ntest = 4000  # Number of instances in the test data
        else:
            self.kappa_file = 'sampled_WIDE12H_bin2_2arcmin_kappa.npy'
            self.label_file = 'sampled_label.npy'
            self.Ncosmo = 3    # Number of cosmologies in the sampled training data
            self.Nsys = 30     # Number of systematic realizations in the sampled training data
            self.test_kappa_file = 'sampled_WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'
            self.Ntest = 3     # Number of instances in the sampled test data

        self.shape = [1424,176] # dimensions of each map
        self.pixelsize_arcmin = 2 # pixel size in arcmin
        self.pixelsize_radian = self.pixelsize_arcmin / 60 / 180 * np.pi # pixel size in radian
        self.ng = 30  # galaxy number density. This determines the noise level of the experiment. Do not change this number.
        self.mask = Utility.load_np(data_dir=self.data_dir, file_name=self.mask_file) # A binary map that shows which parts of the sky are observed and which areas are blocked
        self.label = Utility.load_np(data_dir=self.data_dir, file_name=self.label_file) # Training labels (cosmological and physical paramameters) of each training map
        self.viz_label = Utility.load_np(data_dir=self.data_dir, file_name=self.viz_label_file) # For visualization of parameter distributions

    def load_train_data(self):
        self.kappa = np.zeros((self.Ncosmo, self.Nsys, *self.shape), dtype=np.float16)
        self.kappa[:,:,self.mask] = Utility.load_np(data_dir=self.data_dir, file_name=self.kappa_file) # Training convergence maps

    def load_test_data(self):
        self.kappa_test = np.zeros((self.Ntest, *self.shape), dtype=np.float16)
        self.kappa_test[:,self.mask] = Utility.load_np(data_dir=self.data_dir, file_name=self.test_kappa_file) # Test noisy convergence maps

        
class Visualization:

    @staticmethod
    def plot_mask(mask):
        plt.figure(figsize=(30,100))
        plt.imshow(mask.T)
        plt.show()

    @staticmethod
    def plot_noiseless_training_convergence_map(kappa):
        plt.figure(figsize=(30,100))
        plt.imshow(kappa[0,0].T, vmin=-0.02, vmax=0.07)
        plt.show()

    @staticmethod
    def plot_noisy_training_convergence_map(kappa, mask, pixelsize_arcmin, ng):
        plt.figure(figsize=(30,100))
        plt.imshow(Utility.add_noise(kappa[0,0], mask, ng, pixelsize_arcmin).T, vmin=-0.02, vmax=0.07)
        plt.show()

    @staticmethod
    def plot_cosmological_parameters_OmegaM_S8(label):
        plt.scatter(label[:,0,0], label[:,0,1])
        plt.xlabel(r'$\Omega_m$')
        plt.ylabel(r'$S_8$')
        plt.show()

    @staticmethod
    def plot_baryonic_physics_parameters(label):
        plt.scatter(label[0,:,2], label[0,:,3])
        plt.xlabel(r'$T_{\mathrm{AGN}}$')
        plt.ylabel(r'$f_0$')
        plt.show()

    @staticmethod
    def plot_photometric_redshift_uncertainty_parameters(label):
        plt.hist(label[0,:,4], bins=20)
        plt.xlabel(r'$\Delta z$')
        plt.show()

class Score:
    @staticmethod
    def _score_phase1(true_cosmo, infer_cosmo, errorbar):
        """
        Computes the log-likelihood score for Phase 1 based on predicted cosmological parameters.

        Parameters
        ----------
        true_cosmo : np.ndarray
            Array of true cosmological parameters (shape: [n_samples, n_params]).
        infer_cosmo : np.ndarray
            Array of inferred cosmological parameters from the model (same shape as true_cosmo).
        errorbar : np.ndarray
            Array of standard deviations (uncertainties) for each inferred parameter
            (same shape as true_cosmo).

        Returns
        -------
        np.ndarray
            Array of scores for each sample (shape: [n_samples]).
        """

        sq_error = (true_cosmo - infer_cosmo)**2
        scale_factor = 1000  # This is a constant that scales the error term.
        score = - np.sum(sq_error / errorbar**2 + np.log(errorbar**2) + scale_factor * sq_error, 1)
        score = np.mean(score)
        if score >= -10**6: # Set a minimum of the score (to properly display on Codabench)
            return score
        else:
            return -10**6

def KL_div_posterior_loss(pred_means, pred_sigmas, truths):
    """
    A KL divergence loss function that directly optimizes the score function

    Inputs:
    - pred_means:   2D tensor (batch_size, 2)
    - pred_sigmas:  2D tensor (batch_size, 2)
    - truths:       2D tensor (batch_size, 2)
    """

    residuals_sq = (pred_means - truths)**2

    loss_terms = residuals_sq / (pred_sigmas**2)
    loss_sum = torch.sum(loss_terms, dim=1)

    log_sigma_terms = torch.sum(torch.log(pred_sigmas**2), dim=1)
    loss = torch.mean(loss_sum + log_sigma_terms)

    return loss

class CosmologyDataset(Dataset):
    """
    Custom PyTorch Dataset
    """

    def __init__(self, data, labels=None,
                 transform=None,
                 label_transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].astype(np.float64)   
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx].astype(np.float64)
            label = torch.from_numpy(label)
            if self.label_transform:
                label = self.label_transform(label)
            return image, label
        else:
            return image

if __name__ == "__main__":

    USE_PUBLIC_DATASET = True

    # USE_PUBLIC_DATASET = True
    PUBLIC_DATA_DIR = os.path.join(root_dir, 'input_data/')  # This is only required when you set USE_PUBLIC_DATASET = True

    # %%
    if not USE_PUBLIC_DATASET:                                         # Testing this startking kit with a tiny sample of the training data (3, 30, 1424, 176)
        DATA_DIR = os.path.join(root_dir, 'input_data/')
    else:                                                              # Training your model with all training data (101, 256, 1424, 176)
        DATA_DIR = PUBLIC_DATA_DIR

    # ### Load the train and test data

    # %%
    # Initialize Data class object
    data_obj = Data(data_dir=DATA_DIR, USE_PUBLIC_DATASET=USE_PUBLIC_DATASET)

    # Load train data
    data_obj.load_train_data()

    # %%
    Ncosmo = data_obj.Ncosmo
    Nsys = data_obj.Nsys
    ng = data_obj.ng

    # %%
    # Add the pixel-level noise to the training set (note that this may take some time and large memory)

    # Load clean_kappa and convert to float32 immediately
    # Add the pixel-level noise to the training set (note that this may take some time and large memory)

    np.random.seed(31415)  # Fix the random seed for reproducible results
    clean_kappa = data_obj.kappa.astype(np.float64)
    noisy_kappa = Utility.add_noise(data=clean_kappa,
                                    mask=data_obj.mask,
                                    ng=data_obj.ng,
                                    pixel_size=data_obj.pixelsize_arcmin)
    Utility.save_np(data_dir=DATA_DIR, file_name="noisy_kappa_full.npy",data=noisy_kappa)
    Utility.save_np(data_dir=DATA_DIR, file_name="clean_kappa_full.npy",data=clean_kappa)


    NP_idx = np.arange(Nsys)  # The indices of Nsys nuisance parameter realizations
    split_fraction = 0.2      # Set the fraction of data you want to split (between 0 and 1)
    seed = 5566               # Define your random seed for reproducible results

    train_NP_idx, val_NP_idx = train_test_split(NP_idx, test_size=split_fraction,
                                                random_state=seed)

    noisy_kappa_train = noisy_kappa[:, train_NP_idx]      # shape = (Ncosmo, len(train_NP_idx), 1424, 176)
    clean_kappa_train = clean_kappa[:, train_NP_idx]      # shape = (Ncosmo, len(train_NP_idx), 1424, 176)
    label_train = data_obj.label[:, train_NP_idx]         # shape = (Ncosmo, len(train_NP_idx), 5)
    noisy_kappa_val = noisy_kappa[:, val_NP_idx]          # shape = (Ncosmo, len(val_NP_idx), 1424, 176)
    clean_kappa_val = clean_kappa[:, val_NP_idx]          # shape = (Ncosmo, len(val_NP_idx), 1424, 176)
    label_val = data_obj.label[:, val_NP_idx]             # shape = (Ncosmo, len(val_NP_idx), 5)

    Ntrain = label_train.shape[0]*label_train.shape[1]
    Nval = label_val.shape[0]*label_val.shape[1]

    Utility.save_np(data_dir=DATA_DIR, file_name="noisy_kappa_train.npy",data=noisy_kappa_train)
    Utility.save_np(data_dir=DATA_DIR, file_name="clean_kappa_train.npy",data=clean_kappa_train)
    Utility.save_np(data_dir=DATA_DIR, file_name="label_train.npy",data=label_train)
    Utility.save_np(data_dir=DATA_DIR, file_name="noisy_kappa_val.npy",data=noisy_kappa_val)
    Utility.save_np(data_dir=DATA_DIR, file_name="clean_kappa_val.npy",data=clean_kappa_val)
    Utility.save_np(data_dir=DATA_DIR, file_name="label_val.npy",data=label_val)

    X_train = noisy_kappa_train.reshape(Ntrain, *data_obj.shape)
    X_val = noisy_kappa_val.reshape(Nval, *data_obj.shape)

    # Here, we ignore the nuisance parameters and only keep the 2 cosmological parameters
    y_train = label_train.reshape(Ntrain, 5)[:, :2]
    y_val = label_val.reshape(Nval, 5)[:, :2]

    # %%
    print(f'Shape of the split training data = {X_train.shape}')
    print(f'Shape of the split validation data = {X_val.shape}')

    print(f'Shape of the split training labels = {y_train.shape}')
    print(f'Shape of the split validation labels = {y_val.shape}')
    print(f'All data saved at {DATA_DIR}')
