#!/bin/bash

python exporting.py --train_splits train8 --subsampling_methods OneRandom --subsampling_sizes 100 --epsilons 1e-3 --ref_measure_texts SphereRefMeasure --ref_measure_sizes 1000
#python exporting.py --train_splits train8 --subsampling_methods OneRandom --subsampling_sizes 10000 --epsilons 1e-3 --ref_measure_texts DiskRefMeasure --ref_measure_sizes 1000

#python exporting.py --train_splits train8 --subsampling_methods MultipleRandom --subsampling_sizes 10000 --epsilons 1e-3 --ref_measure_texts SphereRefMeasure --ref_measure_sizes 1000
#python exporting.py --train_splits train8 --subsampling_methods MultipleRandom --subsampling_sizes 10000 --epsilons 1e-3 --ref_measure_texts DiskRefMeasure --ref_measure_sizes 1000
