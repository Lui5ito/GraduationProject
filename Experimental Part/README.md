# Experimental part

This part contains the experimentations of the Sinkhorn kernel on the Rotor37 dataset.
Key files saved here:
- Exploring_Rotor37.ipynb: shows how to import the data, understand the dataset and plot a blade. We want to include statistics about the dataset.
- Optimal_Transport_On_blades.ipynb: performs optimal transport between one blade and multiple reference measure such as another blade, a ball... We also include computation time statistics about performing optimal transport on such huge dataset (one blade is $30,000$ points).
- exporting_sinkhorn.ipynb: performing OT is expensive. This file is simply performing OT with a random blade as reference measure and saving the train and test sample SInkhorn potentials in a .csv file, reusable for the Kernel Ridge Regression.
- trying_reg.ipyng: importing the Sinkhorn potentials .csv file and performing the Kernel Ridge Regression.
- Experiment_On_Real_Dataset.ipynb: big notebook that shows all the steps to perfrom the KRR on blades.

## What is left to do ?
- [x] Analysis of the KRR on train_125.
- Cross-Validation on the kernels parameters.
- Trying different kernels (Matérn...).
- Trying different reference measure.
- Implement the Sliced Wasserstein kernel and compare it to the Sinkhorn kernel. Both in computation time and in performances.
- Explore other existing kernels on 3D-distributions, like the previous kernel introduced by Bachoc in a previous paper.
- Implement Gaussian Process Regression instead of KRR.
- Optimizing the reference measure as shown in the paper. Eventually using the minimum number of sample with maximum performances.