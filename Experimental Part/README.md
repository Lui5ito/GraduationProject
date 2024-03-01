# Experimental part

This part contains the experimentations of the Sinkhorn kernel on the Rotor37 dataset.
Key files saved here:
- Exploring_Rotor37.ipynb: shows how to import the data, understand the dataset and plot a blade. We want to include statistics about the dataset.
- Optimal_Transport_On_blades.ipynb: performs optimal transport between one blade and multiple reference measure such as another blade, a ball... We also include computation time statistics about performing optimal transport on such huge dataset (one blade is $30,000$ points).
- exporting_sinkhorn.ipynb: performing OT is expensive. This file is simply performing OT with a random blade as reference measure and saving the train and test sample Sinkhorn potentials in a .csv file, reusable for the Kernel Ridge Regression. Can use multiple sampling methods.
- exporting_sinkhorn_for_CV.py: Performing the Sinkhorn algorithm for different parameters (train split, epsilon, sampling size and sampling function). The file is .py to enable persistent running on personal computer.
- Performing_KRR.ipyng: importing the Sinkhorn potentials .csv file and performing the Kernel Ridge Regression.
- CrossValidation_KRR.ipyng: importing the Sinkhorn potentials .csv file and cross validates the kernels and regression's parameters using a grid search. Also compare the performance of CV-KRR against train split, epsilon, sampling size and sampling function.
- Experiment_On_Real_Dataset.ipynb: big notebook that shows all the steps to perfrom the KRR on blades.

## What is left to do ?
- [x] Perform KRR on a dataset.
- [x] Cross-Validation on the kernels parameters.
- [x] Sub-sampling the blade using MMD sampling method
  - [ ] Uses too much RAM to perform on T4-Colab GPU's...
  - [ ] Works on personal remote computer. Takes about 1 hour the 100 computation of Optimized Sampling + Sinkhorn Algorithm
- [x] Showing the evolution of the performances with regards to the - this can be seen as hyperparameter tuning. Should all these be selected by cross-validation ? They also impact the computation time, we are looking for a compromise between performance and practicality of use. Currently we are not measuring performances in time.
  - [x] train split
  - [x] sampling size
  - [x] epsilon
  - [x] sampling method - weirdly enough it is not that different between the two... - should we run multiple time for the random sampling ?
  - [ ] reference measure
- [x] Trying different kernels (Matérn...).
- [ ] Trying different reference measure.
- [ ] Implement the Sliced Wasserstein kernel and Mean Maximum Discrepancy and compare it to the Sinkhorn kernel. Both in computation time and in performances. Compare time and memory consumption with Sinkhorn kernel.
- [ ] Explore other existing kernels on 3D-distributions, like the previous kernel introduced by Bachoc in a previous paper.
- [ ] Implement Gaussian Process Regression instead of KRR.
- [ ] Optimizing the reference measure as shown in the paper. Eventually using the minimum number of sample with maximum performances.
- [ ] Implementing an uncertainty quantification of the *efficiency*.

## Questions ❓
- [ ] Is the kernel **really** computing the norm in L2(U) ? The RBF and Matern kernels in scitkitlearn uses distance between two observations. The distance used is the Euclidian distance and therefore we indeed compute the Sinkhorn Kernel
  - [x] Try one kernel for all and look if the results are different -> Results are the same
  - [ ] Try to implement the norm of L2(U) and compute the kernel on that
- [ ] Interpretation of Kernel Ridge Regression
  - [ ] "dual_coefs" gives the $\hat{\alpha}$ from the theoretical problem
  - [ ] How can we quantify the importance of each feature in the regression ?
