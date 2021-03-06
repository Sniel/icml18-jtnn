# Add the current directory so that we can find jtnn module
print("Sys path")
import sys
sys.path.append('')

import torch

from optparse import OptionParser
import rdkit
import rdkit.Chem as Chem
from jtnn import *

lg = rdkit.RDLogger.logger() 
lg.setLevel(rdkit.RDLogger.CRITICAL)

print("Parsing arguments...")
parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path")
parser.add_option("-v", "--vocab", dest="vocab_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=200)
parser.add_option("-l", "--latent", dest="latent_size", default=56)
parser.add_option("-d", "--depth", dest="depth", default=3)
opts,args = parser.parse_args()
   
vocab = [x.strip("\r\n ") for x in open(opts.vocab_path)] 
vocab = Vocab(vocab)

hidden_size = int(opts.hidden_size)
latent_size = int(opts.latent_size)
depth = int(opts.depth)


print("Loading JTNNVAE model...")
model = JTNNVAE(vocab, hidden_size, latent_size, depth)
model.load_state_dict(torch.load(opts.model_path))
model = model.cuda()

data = []
with open(opts.test_path) as f:
    for line in f:
        s = line.strip("\r\n ").split()[0]
        data.append(s)

acc = 0.0
tot = 0
print("Starting reconstruction...")
with open("reconstruction_err.txt", 'w') as f:
    for smiles in data:
        mol = Chem.MolFromSmiles(smiles)
        smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)

        dec_smiles = model.reconstruct(smiles3D)
        if dec_smiles == smiles3D:
            acc += 1
        tot += 1
        print (acc / tot)
        f.write(str(acc/tot) + "\n")
        f.flush()
        """
        dec_smiles = model.recon_eval(smiles3D)
        tot += len(dec_smiles)
        for s in dec_smiles:
            if s == smiles3D:
                acc += 1
        print acc / tot
        """

