# Add the current directory so that we can find jtnn module
import sys

sys.path.append('')

import torch

import numpy as np

from optparse import OptionParser

import rdkit
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols



from jtnn import *
from jtnn import metrics

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

parser = OptionParser()
parser.add_option("-n", "--nsample", dest="nsample")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
opts, args = parser.parse_args()

vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)]
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)
nsample = int(opts.nsample)

model = JTNNVAE(vocab, hidden_size, latent_size, depth)
load_dict = torch.load(opts.model_path)
missing = {k: v for k, v in model.state_dict().items() if k not in load_dict}
load_dict.update(missing)
model.load_state_dict(load_dict)
model = model.cuda()

torch.manual_seed(0)

morgan_samples = []
tanimoto_samples = []
samples = []

for i in range(nsample):
    sample = model.sample_prior(prob_decode=True)
    samples.append(sample)
    print(sample)


data = []
with open("../data/train.txt") as f:
    for line in f:
        s = line.strip("\r\n ").split()[0]
        data.append(s)

print("Internal similarity: {}".format(metrics.internal_diversity(samples)))
print("External similarity: {} ".format(metrics.external_diversity(data, samples)))
