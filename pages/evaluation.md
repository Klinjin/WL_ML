# Evaluation
***
Participants must submit their predictions to the Codabench platform using the test data we provide. The Phase 1 and Phase 2 will be scored independently with two distinct leaderboards, and they will have separate prizes for top participants. 

- ### Phase 1: Cosmological Parameter Estimation
    Participants' models should determine the point estimates $\hat{\Omega}_m$, $\hat{S}_8$ and their one-standard deviation uncertainties $\hat{\sigma}_{\Omega_m}$, $\hat{\sigma}_{S_8}$. The model performance will be ranked with the following score:
        $$
            \begin{aligned}
            & \text { score }_{\text {inference }}=-\frac{1}{N_{\text {test }}} \sum_{i}^{N_{\text {test }}}\left\{\frac{\left(\hat{\Omega}_{m,i}-\Omega_{m,i}^{\text {truth}}\right)^2}{\hat{\sigma}_{\Omega_m,i}^2}+\frac{\left(\hat{S}_{8,i}-S_{8,i}^{\text {truth }}\right)^2}{\hat{\sigma}_{{S_8,i}}^2}\right. \\
            & \left.\quad+\log \left(\hat{\sigma}_{\Omega_m,i}^2\right)+\log \left(\hat{\sigma}_{S_8,i}^2\right)+\lambda\left[\left(\hat{\Omega}_{m,i}-\Omega_{m,i}^{\text {truth }}\right)^2+\left(\hat{S}_{8,i}-S_{8,i}^{\text {truth }}\right)^2\right]\right\}.
            \end{aligned}
        $$
    The first term corresponds to the Kullback–Leibler (KL) divergence (up to some constants) between the true posterior distribution and the Gaussian distribution with the predicted mean and standard deviation. We expect the posterior distribution to be pretty Gaussian and the correlation between $\Omega_m$ and $S_8$ to be small, thus the Gaussian approximation with diagonal covariance matrix should be good enough. The second term is an MSE loss with weight $\lambda=10^3$ to penalize bad point estimates.

    The Phase 1 test data contains 4,000 instances (2D fields similar to images) drawn from the same distribution as the training data with unknown cosmological parameters and 4 systematics.  

- ### Phase 2: Out-of-Distribution Detectionn
    Participants' models should determine the probability $p_i$ that the given dataset is consistent with the training data. The model's OoD detection performance will be assessed with the following score
        $$
            \textrm{score}_{\textrm{OoD}} = \frac{1}{N_{\rm test}} \sum_{i}^{N_{\rm test}} \left[y_i \log(p_i)+(1-y_i)\log(1-p_i)\right],
        $$
    where $y_i=1$ if the dataset is InD, and $y_i=0$ if the dataset is OoD. 

    The Phase 2 test data will contain some instances generated assuming different physical models (OoD). The participants' models should estimate the probability $p_i$ of whether each test data is drawn from the same distribution as the training data. The participant will not be provided with OoD examples or any information on how the OoD test data are generated. 

    We will provide the Phase 2 test data when the Phase 2 starts.