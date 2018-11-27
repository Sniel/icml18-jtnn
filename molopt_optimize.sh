#!/bin/bash
python molopt/optimize.py --test data/opt.test.logP-SA --vocab data/vocab.txt --hidden 420 --depth 3 --latent 56 --sim 0.2 --model molopt/joint-h420-L56-d3-beta0.005/model.iter-4
