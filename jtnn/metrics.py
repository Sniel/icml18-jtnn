import rdkit
import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Fingerprints import FingerprintMols


def internal_diversity(smiles):
    fps = []

    for smile in smiles:
        fps.append(FingerprintMols.FingerprintMol(Chem.MolFromSmiles(smile)))

    internal_divers = 0
    for i, fp1 in enumerate(fps):
        for j, fp2 in enumerate(fps):
            internal_divers += DataStructs.FingerprintSimilarity(fp1, fp2)

    internal_divers /= len(fps) ** 2

    return internal_divers




def external_diversity(train_smiles, generated_smiles):
    train_fps = []
    generated_fps = []

    for train_smile in train_smiles:
        train_fps.append(FingerprintMols.FingerprintMol(Chem.MolFromSmiles(train_smile)))

    for generated_smile in generated_smiles:
        generated_fps.append(FingerprintMols.FingerprintMol(Chem.MolFromSmiles(generated_smile)))

    external_divers = 0
    for i, fp1 in enumerate(train_fps):
        for j, fp2 in enumerate(generated_fps):
            external_divers += DataStructs.FingerprintSimilarity(fp1, fp2)

    external_divers /= (len(train_fps) * len(generated_fps))

    return external_divers
