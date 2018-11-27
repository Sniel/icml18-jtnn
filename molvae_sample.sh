#!/bin/bash
python molvae/sample.py --nsample 10 --vocab data/vocab.txt --model molvae/MPNVAE-h450-L56-d3-noKL/model.iter-2 --hidden 450
