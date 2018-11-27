#!/bin/bash
python molvae/reconstruct.py --test data/train.txt --vocab data/vocab.txt --model molvae/MPNVAE-h450-L56-d3-noKL/model.iter-2 --hidden 450
