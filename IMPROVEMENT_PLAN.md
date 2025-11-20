# WL ML Uncertainty Challenge - Performance Improvement Plan

## Current Performance Analysis

### Model Performance Summary:
- **Simple_CNN Baseline**: 8.27 test score  
- **ResNetWithAttention**: 7.93 test score ❌ (worse than baseline)
- **BigGANUNet2DModel**: 50 min/epoch on CPU ❌ (too slow)
- **Simple_CNN Direct Mean+Sigma 100 epochs**: 7.57 vali score   Ω_m: -7.53%
  S_8: 0.58%
  Overall: -3.48%
- **Small_CNN Direct Mean+Sigma 50 epochs + hparam Optuna**: vali score: 10.02600148418442, :
  Ω_m: -2.89%
  S_8: -0.17%
  Overall: -1.53%
- **SEVER + Small_CNN Direct Mean+Sigma 50 epochs + hparam Optuna**: vali score: 10.02600148418442, :
  Ω_m: -2.89%
  S_8: -0.17%
  Overall: -1.53%

## Root Cause Analysis

### 🚨 Critical Issues with ResNetWithAttention

**1. Massive Batch Size Difference**:
- **ResNetWithAttention**: `BATCH_SIZE = 64` 
- **Simple_CNN baseline**: `BATCH_SIZE = 4`
- **Impact**: 16x larger batch size reduces gradient noise, hurts generalization for small datasets (~26k samples)

**2. Model Complexity vs Dataset Size**:
- **ResNetWithAttention**: 4 ResNet layers + 4 CBAM attention modules + high dropout (0.3, 0.2)
- **Simple_CNN**: Simple 4-layer CNN with minimal dropout (0.2, 0.1)  
- **Impact**: Severe overfitting on limited cosmology training data

**3. Double Precision Overhead**: 
- ResNetWithAttention uses `self.double()` forcing float64
- Simple_CNN uses default float32
- **Impact**: 2x memory usage, slower training, potential numerical issues

### Scoring Function Requirements (`_score_phase1`)
```python
score = - np.sum(sq_error/errorbar**2 + np.log(errorbar**2) + scale_factor * sq_error, 1)
```
This requires:
1. **Accurate predictions** (minimize sq_error)  
2. **Well-calibrated uncertainties** (penalizes overconfident small errorbars)
3. **Balance between accuracy and uncertainty**

### Additional Challenges Identified:
1. **Random noise outliers** in test data (extreme pixel-level Gaussian noise)
2. **Out-of-Distribution (OoD) samples** from different cosmological models or systematic effects
3. **Training data corruption** from nuisance parameters (baryonic physics, photo-z uncertainties)
4. **Direct vs MCMC Inference**: Direct assumes Gaussian posteriors (suboptimal)

---

## 🎯 NEW: Robust Training with SEVER Algorithm

### Overview

The **SEVER (Singular value decomposition for Efficient and Verifiable Error Reduction)** algorithm detects and removes outliers by analyzing the **gradient structure** of the CNN during training. This addresses critical robustness issues in weak lensing parameter estimation.

### Key Insight

SEVER identifies outliers via **SVD of gradient matrix**:
- Outliers create large singular values in specific directions
- Clean data has more uniform gradient structure
- Remove samples with high "outlier scores": $\tau_i = ((\nabla f_i(w) - \bar{\nabla}) \cdot v)^2$

---

## Implementation Roadmap

### Phase 1: Fix ResNetWithAttention Critical Issues (Priority 1) ⚡

**Status**: Needs immediate attention

1. **Fix Batch Size** - MOST CRITICAL:
   ```python
   # Change: BATCH_SIZE = 64  
   # To:     BATCH_SIZE = 4    # Match baseline exactly
   ```

2. **Reduce Model Complexity**:
   - **Remove attention modules** (CBAM adds 4 extra modules)
   - **Reduce ResNet depth**: 2 layers instead of 4 
   - **Lower dropout**: Match Simple_CNN (0.2, 0.1)

3. **Fix Data Type**:
   ```python
   # Remove: self.double()  # Let model use float32
   ```

**Expected Impact**: Score improvement from 7.93 → 8.3+

---

### Phase 2: Core SEVER Implementation (Priority 2) 🛡️

**Timeline**: 1-2 weeks  
**Goal**: Implement gradient-based outlier detection during training

#### 2.1 SEVERFilter Class

Create `SEVERFilter` in `utils.py`:

```python
class SEVERFilter:
    """
    SEVER: Outlier detection via gradient analysis
    
    For weak lensing: Detects maps with anomalous gradients
    that distort cosmological parameter estimates
    """
    
    def __init__(self, model, variance_threshold=1.5, sigma=0.06):
        """
        Args:
            model: Trained CNN (Simple_CNN)
            variance_threshold: Filter threshold (c in Algorithm 2)
            sigma: Approximate learner regularization
        """
        self.model = model
        self.variance_threshold = variance_threshold
        self.sigma = sigma
        
    def compute_gradients(self, dataloader, loss_fn, device):
        """
        Compute gradients for all samples in dataset
        Returns: gradient_matrix [n_samples x n_params]
        """
        self.model.eval()
        all_gradients = []
        
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            X.requires_grad = True
            
            # Forward pass
            pred_means, pred_sigmas = self.model(X)
            loss = loss_fn(pred_means, pred_sigmas, y)
            
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
        # Center gradients
        G = gradient_matrix
        G_centered = G - np.mean(G, axis=0)
        
        # SVD: Find top singular vector (outlier direction)
        U, S, Vt = np.linalg.svd(G_centered, full_matrices=False)
        v = Vt[0]  # Top right singular vector
        
        # Compute outlier scores (Algorithm 1, line 8)
        outlier_scores = ((G_centered @ v) ** 2)
        
        # Filter via variance threshold (Algorithm 2)
        score_variance = np.var(outlier_scores)
        threshold = self.variance_threshold * self.sigma
        
        # Return mask for clean samples (remove top 10% scores)
        inlier_mask = outlier_scores < np.percentile(outlier_scores, 90)
        
        return inlier_mask, outlier_scores
    
    def iterative_filtering(self, dataloader, loss_fn, device, 
                           max_iterations=10):
        """
        Algorithm 1: Iteratively remove outliers until convergence
        """
        current_indices = np.arange(len(dataloader.dataset))
        
        for iteration in range(max_iterations):
            # Compute gradients on current subset
            gradients = self.compute_gradients(dataloader, loss_fn, device)
            
            # Detect outliers
            inlier_mask, scores = self.detect_outliers(gradients)
            
            # Check convergence
            n_removed = (~inlier_mask).sum()
            if n_removed == 0:
                print(f"Converged after {iteration+1} iterations")
                break
                
            print(f"Iteration {iteration+1}: Removed {n_removed} outliers")
            
            # Update dataset indices
            current_indices = current_indices[inlier_mask]
        
        return current_indices, scores
```

#### 2.2 Integration with Training Pipeline

Create `train_direct_robust.py`:

```python
def train_with_sever(model, train_loader, val_loader, config):
    """
    Robust training with SEVER outlier filtering
    """
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=config.LEARNING_RATE,
                                weight_decay=config.WEIGHT_DECAY)
    
    sever_filter = SEVERFilter(model, 
                               variance_threshold=1.5,
                               sigma=config.WEIGHT_DECAY ** 0.5)
    
    best_val_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        # Standard training epoch
        train_loss = train_epoch(model, train_loader, 
                                KL_div_posterior_loss, 
                                optimizer, config.DEVICE)
        
        # Every 5 epochs: filter outliers from training set
        if epoch % 5 == 0 and epoch > 0:
            print(f"\n=== SEVER Filtering at Epoch {epoch} ===")
            clean_indices, scores = sever_filter.iterative_filtering(
                train_loader, KL_div_posterior_loss, config.DEVICE
            )
            
            # Reconstruct training set without outliers
            train_dataset_clean = torch.utils.data.Subset(
                train_loader.dataset, clean_indices
            )
            train_loader = DataLoader(train_dataset_clean, 
                                     batch_size=config.BATCH_SIZE, 
                                     shuffle=True)
            
            print(f"Filtered {len(scores) - len(clean_indices)} outliers")
            print(f"Training set size: {len(clean_indices)}")
        
        # Validation
        val_loss = validate_epoch(model, val_loader, 
                                 KL_div_posterior_loss, config.DEVICE)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    
    return model
```

**Success Criteria**:
- ✅ SEVER filter removes 5-15% of training samples identified as outliers
- ✅ Validation score improves by 0.1-0.2 after filtering
- ✅ Test score ≥ 8.5 (vs 8.27 baseline)

---

### Phase 3: Robust Test-Time Inference (Priority 3) 🎯

**Timeline**: 1 week  
**Goal**: Apply SEVER filtering to test predictions for robust aggregation

#### 3.1 Test Set Outlier Detection

**Challenge**: Test set has **unknown labels** - cannot compute loss gradients directly!

**Solution**: Use **prediction uncertainty** and **feature space anomalies** as proxy:

```python
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
        
        # Project onto top PCs (captures main variation)
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
```

#### 3.2 Ensemble Robust Prediction

**Key Idea**: Multiple predictions with outlier filtering → robust mean/variance

```python
def robust_test_prediction(model, test_loader, config, 
                          n_monte_carlo=10):
    """
    Robust test set prediction with Monte Carlo + SEVER filtering
    
    Strategy:
    1. Generate N predictions with different noise realizations
    2. Filter outlier predictions using SEVER
    3. Aggregate remaining predictions for robust estimates
    """
    test_sever = TestSetSEVER(model, config.DEVICE)
    
    all_predictions = []
    all_uncertainties = []
    
    # Monte Carlo sampling (add different noise each time)
    for mc_iter in range(n_monte_carlo):
        print(f"MC iteration {mc_iter+1}/{n_monte_carlo}")
        
        # Get predictions
        means, sigmas, features = test_sever.compute_prediction_features(
            test_loader
        )
        
        # Detect outliers
        inlier_mask, unc_scores, feat_scores = \
            test_sever.detect_outliers_unsupervised(means, sigmas, features)
        
        all_predictions.append(means)
        all_uncertainties.append(sigmas)
        
        print(f"  Detected {(~inlier_mask).sum()} outliers")
    
    # Aggregate predictions (robust mean)
    all_predictions = np.stack(all_predictions, axis=0)  # [n_mc, n_test, 2]
    all_uncertainties = np.stack(all_uncertainties, axis=0)
    
    # Robust statistics: Use median + MAD instead of mean + std
    final_means = np.median(all_predictions, axis=0)
    final_stds = 1.4826 * np.median(
        np.abs(all_predictions - final_means[None,:,:]), axis=0
    )
    
    # Incorporate model uncertainty
    model_uncertainty = np.mean(all_uncertainties, axis=0)
    final_uncertainties = np.sqrt(final_stds**2 + model_uncertainty**2)
    
    return final_means, final_uncertainties
```

**Success Criteria**:
- ✅ Robust test prediction detects 5-10% outliers in test set
- ✅ Ensemble prediction with SEVER reduces variance by 20-30%
- ✅ Test score ≥ 9.0

---

### Phase 4: SEVER + MCMC Hybrid (Priority 4) 🔬

**Timeline**: 1-2 weeks  
**Goal**: Combine gradient-based outlier detection with Bayesian uncertainty

#### 4.1 MCMC with SEVER-Filtered Likelihood

**Motivation**: Combine robust outlier detection with proper posterior sampling

```python
def mcmc_with_sever(model, test_loader, interpolators, config):
    """
    MCMC sampling with SEVER-filtered likelihood
    
    Workflow:
    1. Get CNN predictions for test samples
    2. Filter outlier predictions using SEVER
    3. Run MCMC only on clean likelihood
    4. Extract posterior statistics
    """
    # Get predictions
    predictions, uncertainties, features = get_predictions(model, test_loader)
    
    # Apply SEVER filtering
    sever = TestSetSEVER(model, config.DEVICE)
    inlier_mask, _, _ = sever.detect_outliers_unsupervised(
        predictions, uncertainties, features
    )
    
    # MCMC sampling (only on inliers)
    posterior_means = []
    posterior_stds = []
    
    for i, (pred, unc) in enumerate(zip(predictions, uncertainties)):
        if not inlier_mask[i]:
            # For outliers: use CNN prediction directly (conservative)
            posterior_means.append(pred)
            posterior_stds.append(unc * 2.0)  # Inflate uncertainty
            continue
        
        # For inliers: Run full MCMC
        samples = run_mcmc_sampling(
            pred, unc, interpolators, n_steps=10000
        )
        
        posterior_means.append(np.mean(samples, axis=0))
        posterior_stds.append(np.std(samples, axis=0))
    
    return np.array(posterior_means), np.array(posterior_stds)
```

**Implementation Steps**:
1. Train Simple_CNN with MSE loss → point estimates
2. Build mean/covariance interpolators from validation data
3. Apply SEVER filtering to identify clean test samples
4. Run MCMC on inliers, use conservative estimates for outliers
5. Extract posterior statistics

**Success Criteria**:
- ✅ SEVER+MCMC hybrid achieves test score ≥ 9.5
- ✅ Better calibrated uncertainties than direct methods
- ✅ Robust to OoD cosmologies

---

## 📊 Evaluation & Validation Strategy

### Metrics to Track

1. **Outlier Detection Performance**:
   - False positive rate (clean samples marked as outliers)
   - True positive rate (actual outliers detected)
   - Impact on validation score after filtering

2. **Robustness to OoD**:
   - Train on subset of cosmologies → test on held-out cosmologies
   - Measure score degradation with/without SEVER

3. **Test Set Performance**:
   - **Primary**: `_score_phase1` on Codabench
   - **Secondary**: Calibration metrics (coverage, sharpness)
   - **Tertiary**: Outlier detection rate on test set

### Controlled Outlier Injection

Create validation sets with known outliers for testing SEVER:

```python
def inject_outliers(X_val, y_val, outlier_fraction=0.1, outlier_type='noise'):
    """
    Create validation set with known outliers for testing SEVER
    """
    n_outliers = int(len(X_val) * outlier_fraction)
    outlier_indices = np.random.choice(len(X_val), n_outliers, replace=False)
    
    X_val_corrupted = X_val.copy()
    
    if outlier_type == 'noise':
        # Add extreme Gaussian noise
        X_val_corrupted[outlier_indices] += np.random.randn(
            *X_val[outlier_indices].shape
        ) * 10.0
    elif outlier_type == 'ood':
        # Replace with samples from different distribution
        X_val_corrupted[outlier_indices] = -X_val[outlier_indices]
    
    return X_val_corrupted, y_val, outlier_indices
```

---

## 🎯 Success Criteria & Expected Impact

### Phase 1 (ResNet Fix):
- **Score: 7.93 → 8.3+** (by fixing batch size and complexity)
- **Timeline**: Immediate (1-2 days)

### Phase 2 (Core SEVER):
- **Score: 8.27 → 8.5+** (training-time outlier filtering)
- **Timeline**: 1-2 weeks
- **Key Deliverable**: `train_direct_robust.py` with SEVERFilter

### Phase 3 (Test-Time Robustness):
- **Score: 8.5 → 9.0** (robust test prediction)
- **Timeline**: 1 week
- **Key Deliverable**: `robust_test_prediction()` function

### Phase 4 (SEVER + MCMC):
- **Score: 9.0 → 9.5+** (Bayesian + robustness)
- **Timeline**: 1-2 weeks
- **Key Deliverable**: `train_HMC.py` with SEVER integration

---

## 💡 Key Insights for Weak Lensing

### Why SEVER Works Here:

1. **Gradient Structure**: Cosmological maps from same $\Omega_m, S_8$ have **similar gradients**
   - Outliers (wrong systematics, extreme noise) produce **anomalous gradients**
   - SVD captures this via large singular values

2. **Nuisance Parameter Marginalization**: 
   - SEVER can filter outliers in nuisance parameter space
   - Focus learning on **cosmology-driven signal**, not systematics

3. **Test-Time Robustness**:
   - Test set noise realizations create pseudo-outliers
   - Filtering before aggregation → robust mean/variance estimates
   - **Critical for Codabench score** (handles extreme noise)

### Adaptations to This Problem:

1. **Use KL divergence loss** (not MSE) for gradient computation
   - Captures uncertainty estimation quality
   - Outliers have large KL divergence gradients

2. **Filter periodically, not every epoch**:
   - Every 5 epochs: Remove ~10% worst samples
   - Allows model to learn from hard examples initially

3. **Conservative test-time filtering**:
   - Don't discard predictions, inflate uncertainties instead
   - Preserves all 4000 test predictions (required for submission)

4. **Combine with domain knowledge**:
   - Physical priors: $\Omega_m \in [0.1, 0.6]$, $S_8 \in [0.6, 1.0]$
   - Filter predictions violating physical constraints

---

## 🛠️ Implementation Timeline

### Week 1-2: Phase 1 & 2
- [ ] Fix ResNetWithAttention critical issues
- [ ] Implement `SEVERFilter` class in `utils.py`
- [ ] Create `train_direct_robust.py`
- [ ] Test on validation set with injected outliers
- [ ] Hyperparameter tuning (variance_threshold, filtering frequency)

### Week 3: Phase 3
- [ ] Implement `TestSetSEVER` for unsupervised detection
- [ ] Create `robust_test_prediction()` function
- [ ] Run Monte Carlo experiments (N=10, 20, 50)
- [ ] Submit to Codabench for scoring

### Week 4-5: Phase 4
- [ ] Integrate SEVER with MCMC pipeline (`train_HMC.py`)
- [ ] Build interpolators from validation data
- [ ] Implement hybrid inference strategy
- [ ] Final hyperparameter optimization

### Week 6: Analysis & Documentation
- [ ] Comprehensive ablation study (with/without SEVER)
- [ ] OoD robustness analysis (held-out cosmologies)
- [ ] Visualize detected outliers (feature space, gradient space)
- [ ] Write technical report

---

## Next Steps (Immediate Actions)

1. **Install optuna** (for hyperparameter tuning):
   ```bash
   pip install --user optuna
   ```

2. **Fix ResNetWithAttention** in `train_direct.py`:
   - Change `BATCH_SIZE = 64` → `BATCH_SIZE = 4`
   - Remove `self.double()` call
   - Simplify architecture

3. **Create new training script**:
   ```bash
   cp train_direct.py train_direct_robust.py
   ```

4. **Implement SEVERFilter class** in `utils.py`

5. **Run pilot experiment**:
   - Train Simple_CNN baseline (already done: 8.27)
   - Train Simple_CNN + SEVER
   - Compare validation scores

---

*Last Updated: November 7, 2025*  
*This plan integrates robust outlier detection (SEVER algorithm) with existing CNN and MCMC approaches for improved weak lensing parameter estimation.*