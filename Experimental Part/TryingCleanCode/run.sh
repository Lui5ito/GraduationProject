#!/bin/bash

# Pour être sur que notre code fonctionne:
#python perform_sinkhorn.py --problem_split train1000 --subsampling_method NoSubsample --subsampling_size 29773 --epsilon 1e-03 --ref_measure disk --ref_measure_size 1000

# Pour comparer avec des epsilon plus grand: 1e-01, 1e-02
#python perform_sinkhorn.py --problem_split train1000 --subsampling_method NoSubsample --subsampling_size 29773 --epsilon 1e-01 --ref_measure disk --ref_measure_size 1000
#python perform_sinkhorn.py --problem_split train1000 --subsampling_method NoSubsample --subsampling_size 29773 --epsilon 1e-02 --ref_measure disk --ref_measure_size 1000

# Pour comparer différentes mesure de référence: disk(10000), disk(100), sphere(1000)
#python perform_sinkhorn.py --problem_split train1000 --subsampling_method NoSubsample --subsampling_size 29773 --epsilon 1e-04 --ref_measure disk --ref_measure_size 100
#python perform_sinkhorn.py --problem_split train1000 --subsampling_method NoSubsample --subsampling_size 29773 --epsilon 1e-04 --ref_measure disk --ref_measure_size 10000
#python perform_sinkhorn.py --problem_split train1000 --subsampling_method NoSubsample --subsampling_size 29773 --epsilon 1e-04 --ref_measure sphere --ref_measure_size 1000

# Pour comparer des sous-échantillonage différents
python perform_sinkhorn.py --problem_split train8 --subsampling_method Optimized --subsampling_size 2000 --epsilon 1e-04 --ref_measure disk --ref_measure_size 1000
python perform_sinkhorn.py --problem_split train8 --subsampling_method OneRandom --subsampling_size 2000 --epsilon 1e-04 --ref_measure disk --ref_measure_size 1000
python perform_sinkhorn.py --problem_split train8 --subsampling_method MultipleRandom --subsampling_size 2000 --epsilon 1e-04 --ref_measure disk --ref_measure_size 1000
python perform_sinkhorn.py --problem_split train8 --subsampling_method OneRandom --subsampling_size 200 --epsilon 1e-04 --ref_measure disk --ref_measure_size 1000
