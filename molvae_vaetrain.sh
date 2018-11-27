#!/bin/bash
python molvae/vaetrain.py --train data/train.txt --vocab data/vocab.txt --save_dir checkpoints
