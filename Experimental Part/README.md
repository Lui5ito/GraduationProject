# Experimental part

## Listes des sujets à évoquer Lundi.

- [ ] **Multioutputs**
  - Le multioutputs, en gros, il fit plusieurs modèles pour chaque Yi (?).
  - Est ce que on peut faire des modèles qui font une seule régression ? Coregionalized kernel ?

- [ ] **Data manquante:** y_test pour efficacité, massflow et ratio de compression.

- [ ] **Modèle de référence et comparaison**
  - Modèle de référence ?
  - Utilisation du noyau Sliced Wasserstein. Quel package ?
  - Utilisation d'un noyau de Bachoc 2020 ?

- [ ] **Mesures de références**
  - Une aube au hasard, il faut bien faire la moyenne des résultats en prenant des aubes différentes ?
  - Une sphère centrée sur les données d'un rayon 'r' ? 
  - Une loi normale 3D ?
  - À chaque fois, quelle taille prendre ?
  - Comment optimiser la mesure de référence ?
     
- [ ] **KRR**
  - Difficile de l'incorporer dans le reste du code à cause du "precomputed" kernel. 

- [ ] **Analyse des modèles**
  - Est ce que il y a un moyen, KRR ou GP, pour analyser la régression. Quelles variables sont les plus influentes ?

- [ ] **Memory Leak**
  - Solutionner le memory leak est ce que c'est le plus important ? Si on veut faire tourner le full Sinkhorn sûrement.

- [ ] **Organisation du code**
  - Comment doit être organisé le code ? Plusieurs fichiers ou un seul ?
  - Comment importer les packages ? Un seul fichier ?

- [ ] **Attendus du projet**
  - Rapport papier ? Modification du premier ou un nouveau ?
  - Présentation du code ? 


## What is left to do ?
- [ ] Faire un fichier functions.py qu'on importe dans un notebook.
  - [ ] ???
- [ ] Faire une fonction qui en créé un nouveau fichier metadata qui stock les performances d'un modèle: temps de Sinkhorn et paramètres, EVS, MSE, temps d'entraînement, temps d'inférence, hyperparametres retenus après cross validation. 
- [ ] Étudier la complexité en stockage de l'algorithme de Sinkhorn. 
- [ ] Solutionner le memory leak.
- [ ] Analyser les résidus
  - [ ] Statistiques: mean, std
  - [ ] Shapiro-Wilk test: check for normality.
  - [ ] Durbin-Watson test: check for autocorrelation.

```
residual_mean = residuals.mean()
residual_std = residuals.std()
print("Mean of residuals:", residual_mean)
print("Standard deviation of residuals:", residual_std)

from scipy.stats import shapiro
statistic, p_value = shapiro(residuals)
print("Shapiro-Wilk test statistic:", statistic)
print("p-value:", p_value)
if p_value > 0.05:
    print("Residuals are normally distributed (fail to reject null hypothesis)")
else:
    print("Residuals are not normally distributed (reject null hypothesis)")

from statsmodels.stats.stattools import durbin_watson
durbin_watson_stat = durbin_watson(residuals)
print("Durbin-Watson test statistic:", durbin_watson_stat)
```


- [ ] Make metadata file for each sinkhorn exportation with sample size, sample méthode, train split, computation time, epsilon, if possible RAM used. How should we store the files ...?
  - Should be organized as a problem first ie train split.
  - Subsampling parameters:
    - Then we have sample method - for the random subsampling, it should be done multiple times? - random, optimized, not subsampled.
    - Sample size - 10, 100, 1000, 2000, 10000.
  - Optimal Transport parameters:
    - Epsilon - Big gap in epsilon should be used 001, 1, 100, 10000
    - Reference measure - every time we randomly select a blade does it mean we should do it multiple times ? It means that we do not use the same reference measure to compare every hyperparameters. 
- [x] Perform KRR on a dataset.
- [x] Cross-Validation on the kernels parameters.
- [x] Sub-sampling the blade using MMD sampling method
  - Uses too much RAM to perform on T4-Colab GPU's...
  - Works on personal remote computer, but runs on CPU.
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
- [x] Implement Gaussian Process Regression instead of KRR. -> Done using the GPy package, manages better the kernels.
- [ ] Optimizing the reference measure as shown in the paper. Eventually using the minimum number of sample with maximum performances.
- [ ] Implementing an uncertainty quantification of the *efficiency*.
- [ ] **Multioutputs models**
  - [ ] How can we implement that in KRR ?
  - [ ] In GP two options: 1. MultiOutputs. 2. Coregionalize.
  - [ ] In Boosting models ?

## Questions ❓
- [ ] Is the kernel **really** computing the norm in L2(U) ? The RBF and Matern kernels in scitkitlearn uses distance between two observations. The distance used is the Euclidian distance and therefore we indeed compute the Sinkhorn Kernel
  - [x] Try one kernel for all and look if the results are different -> Results are the same
  - [ ] Try to implement the norm of L2(U) and compute the kernel on that
- [ ] Interpretation of Kernel Ridge Regression
  - [ ] "dual_coefs" gives the $\hat{\alpha}$ from the theoretical problem
  - [ ] How can we quantify the importance of each feature in the regression ?


## On measuring the quantity of RAM needed
We are using the package [memory_profiler](https://github.com/pythonprofilers/memory_profiler/tree/master). 
We cannot have the plot and the memory consumption for each call of the function.
If you want to have the memory line by line for each call you must have
```
from memory_profiler import profile
```
at the beginning of the file.
If you want to plot nicely with spot for every call you must not use the above line.

How is this package used:
```
mprof run file.py > mporf_outputs.txt 2>&1
```
for computing the sinkhorn potentials and measuring the memory used during each call of the function (enables to isolate each function call).
```
mprof plot -s --output=mprof_plot.png
```
for plotting the time evolution of the RAM.

**Probably** we are going to use it without importing the *memory_profiler* in the .py file. Therfore the plot can be nicely printed and in the *mprofile.dat* file we have the evolution of the memory and markers when the call of a function terminated. To optimize the memory management we are going to *del* every object once they are not used in the function *save_sinkhorn_potentials*.

Shows that we have memory leaks when using Jax. Jax also provides a memory profiler.
- [ ] Understand where is the leak coming from.

## Structure of the saved files for Sinkhorn potentials

```
└── Sinkhorn_Saves
    ├── Split8
        ├── NotSampled
            ├── sinkhorn_potentials_train8_NotSampled_epsilon1.npy
            ├── sinkhorn_potentials_train8_NotSampled_epsilon01.npy
            ├── ...
        ├── OptimizedSample
            ├── Size10
                ├── sinkhorn_potentials_train8_OptimizedSample10_epsilon1.npy
                ├── sinkhorn_potentials_train8_OptimizedSample10_epsilon01.npy
                ├── ...
            ├── Size50
                ├── sinkhorn_potentials_train8_OptimizedSample50_epsilon1.npy
                ├── ...
            ├── Size...
        └── RandomSample
            ├── Size10
                ├── sinkhorn_potentials_train8_RandomSample10_epsilon1.npy
                ├── sinkhorn_potentials_train8_RandomSample10_epsilon01.npy
                ├── ...
            ├── Size50
                ├── sinkhorn_potentials_train8_RandomSample50_epsilon1.npy
                ├── ...
            ├── Size100
                ├── sinkhorn_potentials_train8_RandomSample100_epsilon1.npy
                ├── ...
            ├── Size...
    ├── Split16
        ├── ...
    ├── Split32
        ├── ...
    ├── ...

```
