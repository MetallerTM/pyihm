# **pyIHM**

*pyIHM* is a python software designed in order to offer a comprehensive interface to perform quantitative analyses on NMR spectra of mixtures, using the Indirect Hard Modelling approach.

The Indirect Hard Modelling consists into performing a deconvolution of the spectrum of the mixture using the spectra of the individual components as basis set. Conceptually, the algorithm is made of four steps:
1. fit the spectra of the components of the mixture with a hard model (e.g. Voigt);
2. read and process the spectrum of the mixture;
3. make the initial guess using the set of peaks generated at point 1;
4. get the relative concentrations of the components in the mixture.

The routines for reading and processing of the spectra and for the generation of the models rely on the *KLASSEZ* package.

---

*pyIHM* is developed and tested on *Ubuntu 22.04 LTS* with python 3.11.1. Other OS should encounter no issues; however, if it raises any errors, please notify.
