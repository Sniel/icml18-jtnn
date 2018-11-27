#!/bin/bash
python molopt/vaetrain.py --train data/train.txt --vocab data/vocab.txt --prop data/train.logP-SA --save_dir checkpoints
