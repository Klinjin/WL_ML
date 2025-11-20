#!/usr/bin/env python3
"""
Robust Optuna hyperparameter search with checkpoint/resume capability
Run with: nohup python run_optuna_robust.py > optuna_log.txt 2>&1 &
"""

import os
import json
import sys
import time
import pickle
import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from torchvision import transforms
import optuna

# Import your custom modules
from model import Small_CNN
from robust_estimator import SEVERFilter
from utils import *

# ==============================================================================
# Configuration
# ==============================================================================

ROOT_DIR = os.getcwd()
DATA_DIR = os.path.join(ROOT_DIR, 'input_data/')
MODEL_NAME_PREFIX = "optuna_trial"
CHECKPOINT_FILE = os.path.join(ROOT_DIR, "trained_model/optuna_study_checkpoint.pkl")
BEST_PARAMS_FILE = os.path.join(ROOT_DIR, "trained_model/sever_best_hyperparameters.json")

# Pin to GPU 1
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("="*80)
print(f"ROBUST OPTUNA HYPERPARAMETER SEARCH")
print(f"Started at: {datetime.datetime.now()}")
print("="*80)

# ==============================================================================
# Load Data
# ==============================================================================

print("\nLoading data...")
from utils import Utility

noisy_kappa_train = Utility.load_np(data_dir=DATA_DIR, file_name="noisy_kappa_train.npy")
label_train = Utility.load_np(data_dir=DATA_DIR, file_name="label_train.npy")
noisy_kappa_val = Utility.load_np(data_dir=DATA_DIR, file_name="noisy_kappa_val.npy")
label_val = Utility.load_np(data_dir=DATA_DIR, file_name="label_val.npy")

Ntrain = label_train.shape[0] * label_train.shape[1]
Nval = label_val.shape[0] * label_val.shape[1]

# Reshape
X_train = noisy_kappa_train.reshape(Ntrain, 1424, 176)
X_val = noisy_kappa_val.reshape(Nval, 1424, 176)
y_train = label_train.reshape(Ntrain, 5)[:, :2]
y_val = label_val.reshape(Nval, 5)[:, :2]

print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# Standardization
means = np.mean(X_train, dtype=np.float32)
stds = np.std(X_train, dtype=np.float32)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[means], std=[stds]),
])

label_scaler = StandardScaler()
y_train_scaled = label_scaler.fit_transform(y_train)
y_val_scaled = label_scaler.transform(y_val)

print(f"Image stats: Mean={means:.4f}, Std={stds:.4f}")
print(f"Label stats: Mean={label_scaler.mean_}, Std={np.sqrt(label_scaler.var_)}")

# ==============================================================================
# Dataset and Training Functions
# ==============================================================================

class CosmologyDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx].astype(np.float32)
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx].astype(np.float32)
            label = torch.from_numpy(label)
            return image, label
        else:
            return image


def KL_div_posterior_loss(pred_means, pred_sigmas, truths):
    residuals_sq = (pred_means - truths)**2
    loss_terms = residuals_sq / (pred_sigmas**2)
    loss_sum = torch.sum(loss_terms, dim=1)
    log_sigma_terms = torch.sum(torch.log(pred_sigmas**2), dim=1)
    loss = torch.mean(loss_sum + log_sigma_terms)
    return loss


def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred_means, pred_sigmas = model(X)
        loss = loss_fn(pred_means, pred_sigmas, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate_epoch(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred_means, pred_sigmas = model(X)
            total_loss += loss_fn(pred_means, pred_sigmas, y).item()
    return total_loss / len(dataloader)


def train_with_sever(cnn_model, train_loader, val_loader, config,
                     variance_threshold=1.0, sigma=None, top_fraction=None,
                     max_iterations=100, trial_number=0):
    """
    Simplified SEVER training without visualization (for background execution)
    """
    early_stop_patience = config['EPOCHS'] // 4
    global_best_val_loss = float('inf')
    device = config['DEVICE']
    
    print(f"\n{'='*60}")
    print(f"Trial {trial_number} - SEVER Configuration:")
    print(f"  variance_threshold: {variance_threshold}")
    print(f"  sigma: {sigma}")
    print(f"  top_fraction: {top_fraction}")
    print(f"  max_iterations: {max_iterations}")
    print(f"{'='*60}\n")

    for iteration in range(max_iterations):
        print(f"\n--- Iteration {iteration} ---")
        
        model = cnn_model(
            config['IMG_HEIGHT'],
            config['IMG_WIDTH'],
            config['NUM_TARGETS']
        ).to(device)
        
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['LEARNING_RATE'],
            weight_decay=config['WEIGHT_DECAY']
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        for epoch in range(config['EPOCHS']):
            train_loss = train_epoch(model, train_loader, KL_div_posterior_loss, optimizer, device)
            val_loss = validate_epoch(model, val_loader, KL_div_posterior_loss, device)
            scheduler.step(val_loss)
            
            if epoch % 5 == 0 or val_loss < best_val_loss:
                print(f"  Epoch {epoch+1}/{config['EPOCHS']} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model = model
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        if best_val_loss < global_best_val_loss:
            global_best_val_loss = best_val_loss
            print(f"  *** Global best updated: {global_best_val_loss:.6f}")
        
        # SEVER filtering
        print(f"  Running SEVER filtering...")
        sever_filter = SEVERFilter(
            best_model,
            variance_threshold=variance_threshold,
            sigma=sigma,
            top_fraction=top_fraction
        )
        clean_indices, scores = sever_filter.iterative_filtering(
            train_loader, KL_div_posterior_loss, device
        )
        
        n_filtered = len(train_loader.dataset) - len(clean_indices)
        print(f"  Filtered {n_filtered} outliers | Remaining: {len(clean_indices)}")
        
        # Stopping conditions
        if len(clean_indices) == len(train_loader.dataset) or global_best_val_loss <= -4:
            print(f"  Stopping: no outliers or val_loss <= -4")
            break
        
        # Reconstruct training set
        train_dataset_clean = torch.utils.data.Subset(train_loader.dataset, clean_indices)
        train_loader = DataLoader(
            train_dataset_clean,
            batch_size=config['BATCH_SIZE'],
            shuffle=True
        )
    
    print(f"\nTrial {trial_number} completed | Best val loss: {global_best_val_loss:.6f}")
    return global_best_val_loss


# ==============================================================================
# Optuna Objective with Checkpointing
# ==============================================================================

def objective(trial):
    """Optuna objective function"""
    trial_number = trial.number
    print(f"\n{'#'*80}")
    print(f"# STARTING TRIAL {trial_number}")
    print(f"# Time: {datetime.datetime.now()}")
    print(f"{'#'*80}")
    
    # Hyperparameters
    variance_threshold = trial.suggest_categorical("VARIANCE_THRESHOLD", [0.5, 0.75, 1.0])
    sigma = trial.suggest_categorical("SIGMA", [None, 3e-5**0.5])
    top_fraction = trial.suggest_categorical("TOP_FRACTION", [None, 0.02])
    max_iterations = trial.suggest_categorical("MAX_ITR", [5, 20])
    
    # Config
    config = {
        'IMG_HEIGHT': 1424,
        'IMG_WIDTH': 176,
        'NUM_TARGETS': 4,
        'BATCH_SIZE': 64,
        'EPOCHS': 50,
        'LEARNING_RATE': 4.501717424790536e-05,
        'WEIGHT_DECAY': 3e-5,
        'DEVICE': "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    # Create fresh dataloaders for this trial
    train_dataset = CosmologyDataset(data=X_train, labels=y_train_scaled, transform=transform)
    val_dataset = CosmologyDataset(data=X_val, labels=y_val_scaled, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
    
    try:
        best_val_loss = train_with_sever(
            Small_CNN,
            train_loader,
            val_loader,
            config,
            variance_threshold=variance_threshold,
            sigma=sigma,
            top_fraction=top_fraction,
            max_iterations=max_iterations,
            trial_number=trial_number
        )
        
        print(f"\nTrial {trial_number} SUCCEEDED | Loss: {best_val_loss:.6f}\n")
        return best_val_loss
        
    except Exception as e:
        print(f"\n!!! Trial {trial_number} FAILED !!!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return float('inf')  # Return worst possible value


# ==============================================================================
# Main Execution with Resume Capability
# ==============================================================================

def main():
    # Check for existing study
    if os.path.exists(CHECKPOINT_FILE):
        print(f"\n*** RESUMING from checkpoint: {CHECKPOINT_FILE} ***\n")
        with open(CHECKPOINT_FILE, 'rb') as f:
            study = pickle.load(f)
        print(f"Loaded study with {len(study.trials)} completed trials")
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        print(f"Completed: {completed_trials}, Failed: {len(study.trials) - completed_trials}")
    else:
        print(f"\n*** STARTING NEW STUDY ***\n")
        study = optuna.create_study(
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(),
            study_name="sever_hyperparameter_search"
        )
    
    # Callback to save after each trial
    def save_checkpoint(study, trial):
        with open(CHECKPOINT_FILE, 'wb') as f:
            pickle.dump(study, f)
        print(f"\n>>> Checkpoint saved: {CHECKPOINT_FILE}")
        print(f">>> Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}/20\n")
    
    # Calculate remaining trials
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, 20 - completed)
    
    if remaining == 0:
        print("\n*** ALL 20 TRIALS ALREADY COMPLETED ***\n")
    else:
        print(f"\n*** Running {remaining} remaining trials (out of 20 total) ***\n")
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=remaining,
            timeout=3600 * 10,  # 10 hours
            callbacks=[save_checkpoint]
        )
    
    # Save final results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest trial:")
    print(f"  Value (best_val_loss): {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")
    print(f"  Trial number: {study.best_trial.number}")
    
    best_params_data = {
        "best_val_loss": float(study.best_trial.value),
        "best_params": study.best_trial.params,
        "best_trial_number": study.best_trial.number,
        "total_trials": len(study.trials),
        "completed_at": str(datetime.datetime.now())
    }
    
    os.makedirs(os.path.dirname(BEST_PARAMS_FILE), exist_ok=True)
    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(best_params_data, f, indent=2)
    
    print(f"\nBest hyperparameters saved to: {BEST_PARAMS_FILE}")
    print(f"Checkpoint file: {CHECKPOINT_FILE}")
    print(f"\nFinished at: {datetime.datetime.now()}")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n*** Interrupted by user - progress saved to checkpoint ***")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n!!! FATAL ERROR !!!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
