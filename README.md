# FAIR Universe - Weak Lensing ML Uncertainty Challenge

This repository consists of materials for the Weak Lensing ML Uncertainty Challenge, a NeurIPS 2025 Machine Learning competition that explores uncertainty-aware and out-of-distribution detection AI techniques for **Weak Gravitational Lensing Cosmology**.

The competition is structured into two phases. 
- **The Phase 1 of the competition has started! We are now accepting submissions until November 16th 2025.**
- The Phase 2 of the competition is expected to be launched in **Late September 2025**.

Please check the [**<ins>Competition Overview</ins>**](https://fair-universe.lbl.gov/tutorials/WL_Competition_Overview.pdf) for a high-level overview about this competition, the train/test data structure, evaluation metrics, and the expected competition timeline.

## Competition Website
Interested in joining the competition? Please register as a participant on our [**<ins>Competition Website</ins>**](https://www.codabench.org/competitions/8934/) on Codabench!

***

## Introduction
Weak gravitational lensing reveals the large-scale structure of the universe by observing subtle distortions in galaxy shapes due to intervening matter. Traditional methods mainly capture Gaussian information through two-point statistics. However, non-Gaussian features present in the cosmic web are crucial for studying the underlying physics of structure formation, dark matter distributions, and cosmological parameters, motivating the use of higher-order statistics and advanced machine learning (ML) techniques to extract richer information from the weak lensing data. The primary difficulty is handling systematic uncertainties arising from simulation inaccuracies and observational biases.

Through this competition, participants will analyze a suite of carefully designed mock weak lensing maps with known cosmological parameters, constructed to include variations in simulation fidelity and observational systematics. By comparing the performance and robustness of different methods in a controlled setting, the competition aims to systematically assess their ability to extract cosmological information while quantifying their sensitivity to modeling assumptions and systematics.

The outcomes of this competition are expected to guide the development of next-generation weak lensing analysis pipelines, foster cross-disciplinary collaboration between the astrophysics and machine learning communities, and ultimately improve the reliability of cosmological inference from current and upcoming surveys such as LSST, Euclid, and the Roman Space Telescope. By explicitly addressing simulation-model mismatch and the need to quantify systematic uncertainties, this competition emphasizes scientific robustness and interpretability, aligning with the growing emphasis on trustworthy ML in scientific domains.


## Dataset
Participants will work with simulated datasets mimicking observations from the [Hyper Suprime-Cam (HSC) survey](https://science.jpl.nasa.gov/projects/hyper-suprime-cam/). The weak lensing convergence maps are generated from cosmological simulations with $101$ different cosmological models (parameters: $\Omega_m$ and $S_8$) and realistic systematic effects such as the baryonic effect and photometric redshift uncertainty. These systematics are introduced in the data generation process, which we fully sampled in the training set so that the participants can marginalize over them. The parameters corresponding to these systematic models are nuisance parameters and need to be marginalized during inference. Each data is a 2D image of dimension $1424 \times 176$, corresponds to the convergence map of redshift BIN 2 of WIDE12H in HSC Y3, pixelized with a resolution of 2 arcmin. 

The figure below shows some examples of the training data and how they are varied with different nuisance parameters and pixel-level noise.

<center>
<img src="image-1.png" width="600">
</center>

## Competition Tasks
### Phase 1: Cosmological Parameter Estimation
Participants will develop models that:
1. Accurately infer cosmological parameters $(\hat{\Omega}_m, \hat{S}_8)$ from the weak lensing data.
2. Quantify the uncertainties $(\hat{\sigma}_{\Omega_m}, \hat{\sigma}_{S_8})$ via the 68% confidence intervals of the estimated cosmological parameters.

### Phase 2: Out-of-Distribution Detection
Some test data will be generated with different physical models (OoD), leading to some distribution shifts with respect to the test data in Phase 1. Participants will develop models that:
1. Identify test data samples inconsistent with the training distribution (OoD detection).
2. Provide probability estimates indicating data conformity to training distributions.
   

## Getting Started
### Phase 1: Cosmological Parameter Estimation
We have prepared a Starting Kit for the Phase 1 competition [<ins>here</ins>](https://github.com/FAIR-Universe/Cosmology_Challenge/blob/master/Phase_1_Startingkit_WL_PSAnalysis.ipynb). You can also directly run the starting kit on Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iySb87VmyCtz6y8Zg367xR6hetD6gKTi?usp=sharing)

The notebook includes a code example for data loading, baseline approach, evaluation, and submission preparation. The Phase 1 baseline apporach is the standard power spectrum analysis. Other baseline methods employing basic CNN emulators will be available soon.

### Phase 2: Out-of-Distribution Detection
The Starting Kit with a baseline approach for Phase 2 will be available when the Phase 2 starts.

## Contact
Visit our website: https://fair-universe.lbl.gov/

Email: fair-universe@lbl.gov
