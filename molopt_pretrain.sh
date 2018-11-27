#!/bin/bash
python molopt/pretrain.py --train data/train.txt --vocab data/vocab.txt --prop data/train.logP-SA --save_dir checkpoints
