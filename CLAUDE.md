# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the **FAIR Universe - Weak Lensing ML Uncertainty Challenge** repository for the NeurIPS 2025 competition. The challenge focuses on uncertainty-aware and out-of-distribution detection AI techniques for weak gravitational lensing cosmology.

## Key Architecture Components

### Core Machine Learning Models
- **BigGANUNet2DModel** (`model.py`): Advanced U-Net architecture with BigGAN components, supports Fourier features and self-attention
- **ResNetWithAttention** (`model.py`): ResNet-based model with attention mechanisms for cosmological parameter estimation
- **LayersPP** (`layerspp.py`): Custom neural network layers with advanced initialization and normalization

### Data Processing Pipeline
- **Data class** (`utils.py`): Handles loading and preprocessing of weak lensing convergence maps (1424×176 pixels)
- **Utility class** (`utils.py`): Noise addition, data loading/saving, evaluation metrics computation
- Supports both public dataset (101 cosmological models, 256 realizations each) and smaller test datasets

### Training Scripts
- **train_direct.py**: Direct parameter prediction using KL divergence loss (ResNetWithAttention)
- **train_HMC.py**: **NEW** - Advanced MCMC/HMC inference with Simple_CNN backbone
  - **MCMC mode**: Metropolis-Hastings sampling (original CNN_MCMC.ipynb approach)
  - **HMC mode**: Hamiltonian Monte Carlo using numpyro.infer.NUTS
  - Better uncertainty quantification than direct methods
  - Handles parameter correlations and non-Gaussian posteriors

## Competition Structure

### Phase 1: Cosmological Parameter Estimation
- Infer cosmological parameters (Ωₘ, S₈) from weak lensing convergence maps
- Quantify uncertainties via 68% confidence intervals
- Handle systematic effects (baryonic effects, photometric redshift uncertainty)

### Phase 2: Out-of-Distribution Detection
- Identify test samples inconsistent with training distribution
- Provide probability estimates for data conformity

## Development Commands

### Environment Setup
```bash
# Create conda environment
conda create --name WL_ML_Challenge python=3.12.8
conda activate WL_ML_Challenge
pip install -r conda/requirements.txt
```

### Training Models
```bash
# Direct parameter prediction (ResNetWithAttention)
python train_direct.py

# MCMC/HMC inference (Simple_CNN + posterior sampling)
python train_HMC.py                                    # Default: MCMC with public dataset
python train_HMC.py --method hmc                       # Use HMC inference
python train_HMC.py --method mcmc --model-name Custom  # Custom model name
python train_HMC.py --no-public-dataset               # Use small test dataset

# SLURM job submission (for HPC environments)
sbatch job.sh
```

### Command Line Options for train_HMC.py
- `--method {mcmc,hmc}`: Inference method (default: mcmc)
- `--use-public-dataset`: Use full public dataset (default: True)
- `--no-public-dataset`: Use small test dataset instead
- `--model-name NAME`: Custom name for model and output files

### Jupyter Notebooks
- `Phase_1_Startingkit_WL_CNN_Direct.ipynb`: CNN direct prediction baseline
- `Phase_1_Startingkit_WL_PSAnalysis.ipynb`: Power spectrum analysis baseline
- `CNN_MCMC.ipynb`: CNN + MCMC approach
- `ResNet_Direct.ipynb`: ResNet-based direct prediction

## Key Dependencies
- PyTorch (main ML framework)
- NumPy, Matplotlib (data handling/visualization)
- scikit-learn (preprocessing, metrics)
- tqdm (progress bars)

## Competition Framework
- **Ingestion Program** (`ingestion_program/`): Handles model execution and prediction generation
- **Scoring Program** (`scoring_program/`): Evaluates submissions using R² scores and other metrics
- **Sample Submissions** (`sample_result_submission/`): Reference format for competition submissions

## Data Structure
- Input data: 2D convergence maps (1424×176 pixels, 2 arcmin resolution)
- Labels: Cosmological parameters (Ωₘ, S₈) plus nuisance parameters
- Training: 101 cosmological models × 256 systematic realizations
- Galaxy density assumption: varies per experiment, affects noise level

## HPC Configuration
The repository includes SLURM job scripts configured for:
- CPU-only training (128 cores, 20+ hour jobs)
- JAX backend configuration for CPU optimization
- Environment loading from virtual environments

## File Organization
- Core models: `model.py`, `layerspp.py`, `layers.py`
- Training utilities: `utils.py`, `train_*.py`
- Competition infrastructure: `ingestion_program/`, `scoring_program/`
- Data directories: `input_data/`, `trained_model/`, `scoring_output/`