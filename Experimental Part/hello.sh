#!/bin/bash

python exporting_sinkhorn.py --train_splits train1000 --subsampling_methods NotSampled --subsampling_sizes 29773 --epsilons 1e-3 --ref_measure_texts SphereRefMeasure --ref_measure_sizes 1000
python exporting_sinkhorn.py --train_splits train1000 --subsampling_methods NotSampled --subsampling_sizes 29773 --epsilons 1e-3 --ref_measure_texts DiskRefMeasure --ref_measure_sizes 1000

python exporting_sinkhorn.py --train_splits train1000 --subsampling_methods NotSampled --subsampling_sizes 29773 --epsilons 1e-4 --ref_measure_texts SphereRefMeasure --ref_measure_sizes 1000
python exporting_sinkhorn.py --train_splits train1000 --subsampling_methods NotSampled --subsampling_sizes 29773 --epsilons 1e-4 --ref_measure_texts DiskRefMeasure --ref_measure_sizes 1000
