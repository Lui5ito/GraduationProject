# Graduation Project

This repo contains my Master's end-of-year project, completed during my studies at ENSAI. It aims at reproducing experiments from the article "Gaussian Processes on Distributions based on Regularized Optimal Transport" and experiment on publicly available dataset.

The article is available at: [Gaussian Processes on Distributions based on Regularized Optimal Transport](https://arxiv.org/abs/2210.06574)

The first part of the project consisted of a *methododogical* part. The goal was to familiarize with the paper, the Optimal Transport theory and reproduce one of the experiment of the paper. Our written report, slides and Jupyter notebook can be found [here](https://github.com/Lui5ito/GraduationProject/tree/main/Methodological%20Part).

The second part of the project, *experimental* part, is about applying the foundings of the paper on a real life dataset and compare the proposed method against established ones such as the Wasserstein distance. We explored the [Rotor37](https://plaid-lib.readthedocs.io/en/latest/source/data_challenges/rotor37.html) dataset and our notebooks and codes are [here](https://github.com/Lui5ito/GraduationProject/tree/main/Experimental%20Part).

| <img src="https://github.com/Lui5ito/GraduationProject/assets/104061901/076f6dc7-0fec-4b08-8823-244610d82705" alt="image" width="150" height="auto">  |  <img src="https://github.com/Lui5ito/GraduationProject/assets/104061901/dac8126d-2854-4b9f-929f-cbae6dca4cf7" alt="image" width="150" height="auto">  |   <img src="https://github.com/Lui5ito/GraduationProject/assets/104061901/37bd279e-9860-447b-8c73-aa59151c7f01" alt="image" width="150" height="auto">  | 
|:-:|:-:|:-:|
| A blade | Disk as reference measure | Uniform sphere as reference measure |


## Main results

Our report exhibit the performance of the Sinkhorn kernel presented in the paper on the Rotor37 dataset.

We found that the Sinkhorn kernel performs really well to predict the efficency, the massflow and the compression ratio of the blade.

| <img src="https://github.com/Lui5ito/GraduationProject/assets/104061901/675c42de-0029-40f0-bc6c-2cfe88ca1391" alt="image" width="200" height="auto">  |  <img src="https://github.com/Lui5ito/GraduationProject/assets/104061901/657ca77b-000a-4168-ba2e-4bc19c616b4b" alt="image" width="200" height="auto">  |   <img src="https://github.com/Lui5ito/GraduationProject/assets/104061901/e988d141-8cf2-4638-bf18-0ec2ccd4c481" alt="image" width="200" height="auto">  | 
|:-:|:-:|:-:|
| Efficiency, $EVS = 0.9895$ | Massflow, $EVS = 0.9983$ | Compression ratio, $EVS = 0.9983$ |

We also explore multiple hyperparameters. Here are our main findings:
- There exist an epsilon from which the performance of the regression task are great. But going with even smaller epsilon do not seem to impact the model.
- One do not need much points in the reference measure, a few hundreds is enough. Therefore the computation is not that big even for huge datasets.
