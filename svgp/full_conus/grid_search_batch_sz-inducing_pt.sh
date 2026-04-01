#!/bin/bash
for bs in 16384 32768 49152; do
    python -u loso_cv_svgp_parallel.py --n_gpus 4 --n_inducing 256,512,1024,2048,4096 \
        --n_epochs 50 --n_sites 5 --states MT --batch_size $bs
done
