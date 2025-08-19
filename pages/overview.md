# Overview 
*** 
## Introduction
The large-scale structure of the universe—the cosmic web of galaxies, galaxy clusters, and dark matter spanning hundreds of millions of light-years—encodes essential information about the composition, evolution, and fundamental laws governing the cosmos. However, the majority of matter in the universe is dark matter, which does not interact with light and can only be observed indirectly through its gravitational effects. According to Einstein’s theory of general relativity, the gravitational field of this large-scale structure bends the path of light traveling through the universe. **Weak gravitational lensing** refers to the subtle, coherent distortions in the observed shapes of distant galaxies caused by the deflection of light as it traverses the inhomogeneous matter distribution of the universe. By statistically analyzing these distortions across large regions of the sky, weak lensing provides a powerful probe of the matter distribution and the underlying cosmological model that governs the expansion of the universe.

Traditional analysis based on two-point correlation functions can only capture limited amount of information from the weak lensing data (2D fields similar to images). To fully exploit the non-Gaussian features present in the cosmic web, higher-order statistics and modern machine learning (ML) methods have become increasingly important. These approaches, including deep learning and simulation-based inference, have been shown to extract significant more information in weak lensing maps than traditional techniques. However, different analyses assume different dataset setups and lead to different results, making it hard to directly compare with existing approaches. Furthermore, most (if not all) of these methods rely heavily on simulations that may not accurately represent real data due to modeling approximations and missing systematics. 


This challenge is motivated by the need to quantify and compare the information content that different analysis methods—ranging from classical statistics to ML-based models—can extract from weak lensing maps, while also evaluating their robustness to simulation inaccuracies and observational systematics.



The outcomes of this challenge are expected to guide the development of next-generation weak lensing analysis pipelines, foster cross-disciplinary collaboration between the astrophysics and machine learning communities, and ultimately improve the reliability of cosmological inference from current and upcoming surveys such as LSST, Euclid, and the Roman Space Telescope. By explicitly addressing simulation-model mismatch and the need to quantify systematic uncertainties, this challenge emphasizes scientific robustness and interpretability, aligning with the growing emphasis on trustworthy ML in scientific domains.


## Challenge summary: 
***
Through this challenge, participants will analyze a suite of carefully designed mock weak lensing maps with known cosmological parameters, constructed to include variations in simulation fidelity and several observational systematic uncertainties. By comparing the performance and robustness of different methods in a controlled setting, the challenge aims to systematically assess their ability to extract cosmological information while quantifying their sensitivity to modeling assumptions and systematics.

In practice, the participant will be provided with a large labeled training dataset ($\sim$ 26k) to train models that learn useful features for cosmological inference on two key parameters:
- $\Omega_m$: a parameter that represents the fraction of the matter energy density in the universe.
- $S_8$: a parameter that quantifies the amplitude of matter fluctuations in the present-day universe.

This challenge tasks are structured in two phases:
- **First phase:** 
Participants are tasked with developing models that can accurately infer key cosmological parameters $\Omega_m$ and $S_8$ from a dataset designed to mimic weak lensing observations. However, due to limitations in our simulations and the modeling of various systematic effects, there may be a mismatch between the simulated data and real observations. This simulation-model mismatch, or distribution shift, can introduce significant biases in parameter inference. 
Participants' models should predict both point estimates $(\hat{\Omega}_m, \hat{S}_8)$ and their corresponding one-standard deviation uncertainties $(\sigma_{\Omega_m},\sigma_{S_8})$. 
The point estimate and the uncertainties could be obtained by, for example, sampling the posterior with Markov chain Monte Carlo (MCMC), or from a Maximum Likelihood Fit estimator. 

- **Second phase:** 
Participants have to develop methods for out-of-distribution (OoD) detection, with the goal of identifying the test data that deviates from the training distribution through the OoD probability. 
This OoD probability $p$ can be potentially obtained by, for example, estimating the probability distribution of training data or evaluating some distance metrics between test and training data in a learned feature space.

Our training datasets incorporate all major known systematics and are constructed to be as realistic as possible. As a result, we anticipate that models developed through this challenge will be directly applicable to real observational data, enabling more robust and precise cosmological measurements.


## How to join this challenge?
***
- Go to the "Starting Kit" tab
- Download the "Dummy sample submission"
- Go to the "My Submissions" tab
- Submit the downloaded file


## Submissions
***
This competition allows only result submissions. Participants can submit a result submission as instructed in the Starting Kit tab.

## Data
***
We have prepared training data for participants that can be downloeded from [Here](https://www.codabench.org/datasets/download/c99c803a-450a-4e51-b5dc-133686258428/)


## Credits
***
#### Institute for Advanced Study
- Biwei Dai 

#### Lawrence Berkeley National Laboratory 
- Wahid Bhimji
- Paolo Calafiura
- Po-Wen Chang
- Sascha Diefenbacher
- Jordan Dudley
- Steven Farrell
- Chris Harris
- Benjamin Nachman
- Uroš Seljak


#### University of Washington
- Yuan-Tang Chou
- Elham E Khoda
- Yulei Zhang

#### ChaLearn
- Isabelle Guyon
- Ihsan Ullah

#### Université Paris-Saclay
- Ragansu Chakkappai
- David Rousseau



## Contact
***
Visit our website: https://fair-universe.lbl.gov/

Email: fair-universe@lbl.gov

Updates will be announced through fair-universe-announcements google group. [Click to join Google Group](https://groups.google.com/u/0/a/lbl.gov/g/Fair-Universe-Announcements/)