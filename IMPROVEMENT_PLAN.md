# WL ML Uncertainty Challenge - Performance Improvement Plan

## Current Performance Analysis

### Model Performance Summary:
- **Simple_CNN Baseline**: 8.27 test score  
- **ResNetWithAttention**: 7.93 test score ❌ (worse than baseline)
- **BigGANUNet2DModel**: 50 min/epoch on CPU ❌ (too slow)

## Root Cause Analysis (CORRECTED)

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

### Secondary Issues:
1. **Model Complexity vs Data Size**: ResNet+Attention overfits on ~26k samples
2. **Direct vs MCMC Inference**: Direct assumes Gaussian posteriors (suboptimal)  
3. **Computational Constraints**: CPU-only training limits complex architectures
4. **Hyperparameter Inconsistency**: Different optimizers, batch sizes between models

## Immediate Action Plan

### Phase 1: Fix ResNetWithAttention Critical Issues (Priority 1) ⚡

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

4. **Architecture Simplification**:
   - Start with Simple_CNN architecture but add **1-2 residual connections**
   - Add **minimal attention** (1 module max, not 4)
   - Keep same hyperparameters as baseline

### Phase 2: Implement CNN_MCMC (Priority 2) 🎯
1. **Rewrite `train_HMC.py`** based on `CNN_MCMC.ipynb`:
   - Use Simple_CNN architecture (proven to work)
   - Implement full MCMC pipeline for uncertainty quantification
   - Compare performance against direct methods

2. **MCMC Implementation Steps**:
   - Train CNN to predict point estimates (MSE loss)
   - Calculate mean/covariance interpolators from training data
   - Use MCMC sampling for posterior exploration
   - Extract mean and std from samples

### Phase 3: Systematic Architecture Comparison (Priority 3) 📊  
1. **Fair Comparison Framework**:
   - Same loss function (KL divergence) for all models
   - Same hyperparameters and training setup
   - Compare: Simple_CNN, Reduced ResNet, Custom CNN

2. **CPU-Optimized Architectures**:
   - Design efficient CNNs for CPU training
   - Focus on depth vs width trade-offs
   - Profile training time vs performance

## Expected Impact

### Quick Wins (Phase 1):
- **ResNetWithAttention score: 7.93 → 8.3+** (by fixing batch size and complexity)
- **Batch size fix alone**: Likely +0.2-0.4 score improvement  
- **Remove overfitting**: Better generalization to test set
- **Fair architecture comparison**: Same training conditions

### Medium-term (Phase 2): 
- **CNN_MCMC approach: 8.3+ → 9.0+** (better uncertainty quantification)
- More principled uncertainty estimates via MCMC sampling
- Better handling of non-Gaussian posteriors and parameter correlations

### Long-term (Phase 3):
- **Systematic architecture optimization for cosmology data**
- **CPU-optimized efficient models**
- **Test scores > 9.5** with optimized architectures

## Key Implementation Notes

### For `train_HMC.py` Implementation:

**Step 1: Copy CNN_MCMC.ipynb structure but use Simple_CNN**
```python
# Use Simple_CNN architecture (proven to work)
model = Simple_CNN(config.IMG_HEIGHT, config.IMG_WIDTH, 2)  # NUM_TARGETS = 2 (not 4!)

# Train with MSE loss (not KL divergence) for point estimates
loss_fn = nn.MSELoss()
```

**Step 2: Critical MCMC Implementation Details**:
1. **Interpolators for Gaussian likelihood**:
   ```python
   from scipy.interpolate import LinearNDInterpolator
   mean_d_vector_interp = LinearNDInterpolator(cosmology, mean_d_vector)
   cov_d_vector_interp = LinearNDInterpolator(cosmology, cov_d_vector)
   ```

2. **MCMC Parameters** (copy exactly from CNN_MCMC.ipynb):
   - `Nstep = 10000` (MCMC steps)
   - `sigma = 0.06` (proposal std) 
   - Burn-in: Remove first 20% of samples
   - Track acceptance rates (target: 20-50%)

**Step 3: Key Implementation Sequence**:
1. Train Simple_CNN with MSE loss → point estimates
2. Group validation predictions by cosmology  
3. Calculate mean/covariance per cosmology
4. Create interpolators for likelihood  
5. Run MCMC sampling for each test sample
6. Extract mean ± std from posterior samples

### Success Metrics:
- **Validation score > 8.27** (beat Simple_CNN baseline)
- **Test score > 7.93** (beat current ResNet)  
- **Training time < 2 hours** (practical on CPU)
- **Well-calibrated uncertainties** (coverage analysis)

## Next Steps

1. **Immediate**: Fix ResNetWithAttention loss function in `train_direct.py`
2. **This week**: Implement `train_HMC.py` with CNN_MCMC approach
3. **Next week**: Systematic architecture comparison and optimization

---

*This analysis was conducted by examining the scoring function (`_score_phase1`), comparing baseline vs current models, and identifying the critical loss function mismatch that explains the performance degradation.*