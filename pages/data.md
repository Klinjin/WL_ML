# Data
***

## Dataset
Participants will work with simulated datasets mimicking observations from the [<ins>Hyper Suprime-Cam (HSC) survey</ins>](https://science.jpl.nasa.gov/projects/hyper-suprime-cam/). Each data is a 2D image of dimension $1424 \times 176$, corresponds to the convergence map of redshift BIN 2 of WIDE12H subfield in HSC Y3, pixelized with a resolution of 2 arcmin. 

These weak lensing convergence maps are generated from high-resolution cosmological ray-tracing simulations with $101$ different spatially-flat $\Lambda \text{CDM}$ cosmological models. Each cosmological model differs in cosmological parameters $\Omega_m$, the fraction of the total matter density of the Universe, and $S_8$, the amplitude of matter fluctuations on $8 \,\mathrm{Mpc}/h$ scales in the Universe today. These two parameters serve as the label of each data. 

In addition to the cosmological signal, we also model various realistic systematic effects (distortions to the data), such as baryonic effect and photometric redshift uncertainty. These systematics are introduced in the data generation process, which we fully sampled in the training set so that the participants can marginalize over them. The parameters corresponding to these systematic models are nuisance parameters and need to be marginalized during inference.

We have prepared the training data and the Phase 1 test data for participants. Please download them from
[**<ins>Training Data / Phase 1 Test Data (6.7 GB)</ins>**](https://www.codabench.org/datasets/download/c99c803a-450a-4e51-b5dc-133686258428/). The Phase 2 test data will be available when the Phase 2 starts.


The figure below shows some examples of the training data and how they are varied with different nuisance parameters and pixel-level noise.
<center>
<img src="image-1.png" width="600"> 
</center>

## Baseline Methods
- ### Phase 1: Cosmological Parameter Estimation

    1. **Power Spectrum Analysis** 

        In cosmology, the power spectrum describes how matter is distributed across different size scales in the universe and is a key tool for studying the growth of cosmic structure. Starting from the matter density $\delta(x)$, we transform it into Fourier space to get $\tilde{\delta}(x)$, which represents fluctuations as waves of different wavelengths. The matter power spectrum P(k) is then defined by:

        $$\langle \tilde{\delta}(\mathbf{k}) \tilde{\delta}^*(\mathbf{k}') \rangle = (2\pi)^3 \delta_D(\mathbf{k}-\mathbf{k}') P(k),$$

        where k is the wavenumber corresponding to a scale $\lambda \sim 1/k$, and $ \delta_D$ is the Dirac delta function. Intuitively, P(k) tells us how "clumpy" the universe is on different scales. In cosmology, the shape and amplitude of P(k) encodes the physics and composition of the universe, making it one of the most important statistical tools in the field.

        In this baseline method, we use power spectrum as the summary statistic to constrain the cosmological parameters, so
        $$
        \boldsymbol{d} = \text{log} ~P(k)~~\text{ with $n_d=$ number of bins in wavenumber } k.
        $$

    2. **Convolutional Neural Network + MCMC**

        In this notebook, we use the outputs of the convolutional neural network (the point estimates of cosmological parameters $\boldsymbol{\theta} = (\hat{\Omega}_m, \hat{S}_8)$) as the summary statistic to constrain the cosmological parameters, so
        $$
        \boldsymbol{d} = f_{\rm NN}^{\phi} ~~\text{ with $n_d = 2$.}
        $$

        The model is trained with an MSE loss.

    3. **Convolutional Neural Network Direct Prediction**

        In this notebook, we do not fit any predefined summary statistics to the data and perform MCMC. Instead, we estimate the uncertainties directly using the CNN. This is achieved by optimizing a KL divergence objective function using neural network predictions during training. For each 2D map, the CNN will predict its cosmological parameters $(\hat{\Omega}_m, \hat{S}_8)$ and the standard deviations of the joint Gaussian posterior distribution $(\hat{\sigma}_{\Omega_m}, \hat{\sigma}_{S_8})$.

        The model is trained with a KL divergence objective function defined by
        $$
        \text{KL Loss}= \frac{1}{N} \sum_i^{N}\left\{\frac{\left(\hat{\Omega}_{m, i}-\Omega_{m, i}^{\text {truth }}\right)^2}{\hat{\sigma}_{\Omega_m, i}^2}+\frac{\left(\hat{S}_{8, i}-S_{8, i}^{\text {truth }}\right)^2}{\hat{\sigma}_{S_8, i}^2}+\log \left(\hat{\sigma}_{\Omega_m, i}^2\right)+\log \left(\hat{\sigma}_{S_8, i}^2\right)\right\}~.
        $$

    The plot below shows a comparison between the sampled posterior distributions of our three baseline methods. Comparing to the traditional power spectrum analysis, the neural networks can capture more information from the weak lensing data, leading to better predictions. 
    <center>
    <img src="image.png" width="400"> 
    </center>

    You can also find visualizations about the dataset and the Phase 1 baseline methods in the `Starting Kit` tab.

- ### Phase 2: Out-of-Distribution Detectionn
    The Phase 2 baseline method and its starting kit will be available when the phase 2 starts.