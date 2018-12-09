import rdkit.Chem as Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
import networkx as nx
import gmatch4py
import jtnn .utils as utils
import jtnn.chemutils
import time

# Based on https://arxiv.org/pdf/1708.08227.pdf
# Compute the internal chemical diversity within a set of molecules: checks how different the molecules are from one another
def internal_diversity(smiles):
    fps = []

    for smile in smiles:
        fps.append(FingerprintMols.FingerprintMol(Chem.MolFromSmiles(smile)))

    internal_divers = 0
    for i, fp1 in enumerate(fps):
        if i % 10000 == 0:
            print(i)
        for j, fp2 in enumerate(fps):
            internal_divers += (1-DataStructs.FingerprintSimilarity(fp1, fp2))

    internal_divers /= len(fps) ** 2

    return internal_divers

# Based on https://arxiv.org/pdf/1708.08227.pdf
# Compute the external chemical diversity: how different are the generated molecules from the train set
def external_diversity(train_smiles, generated_smiles):
    train_fps = []
    generated_fps = []

    for train_smile in train_smiles:
        train_fps.append(FingerprintMols.FingerprintMol(Chem.MolFromSmiles(train_smile)))

    for generated_smile in generated_smiles:
        generated_fps.append(FingerprintMols.FingerprintMol(Chem.MolFromSmiles(generated_smile)))

    external_divers = 0
    for i, fp1 in enumerate(train_fps):
        if i % 10000 == 0:
            print(i)
        for j, fp2 in enumerate(generated_fps):
            external_divers += (1-DataStructs.FingerprintSimilarity(fp1, fp2))

    external_divers /= (len(train_fps) * len(generated_fps))

    return external_divers


def add_molecules(mols, train, result):
    # The first row contains the distance between the first molecules and all other molecules.
    # We remove the first item of the row because it is the same molecule
    result = result[0, 1:]
    for i in range(1, result.shape[0]):
        mols.append((result[i], train[i]))

    return mols


def get_closest_molecules_ged(batch_size, train, generated_smile, n):
    generated_graph = [nx.MultiGraph(jtnn.chemutils.to_graph(generated_smile)[1])]

    ged = gmatch4py.GraphEditDistance(1, 1, 1, 1)
    #ged = gmatch4py.HED(1, 1, 1, 1)
    closest_molecules = []
    train_graph = []
    for i, smiles in enumerate(train):
        if i > 0 and i % 10000 == 0:
            print("Converting smiles to graph... {}".format(i))
        train_graph.append(nx.MultiGraph(jtnn.chemutils.to_graph(smiles)[1]))

    num_batches = int(len(train) / batch_size)
    last_batch_size = len(train) - num_batches * batch_size

    start = time.time()
    # Separate in batches because it speeds up the calculations
    for i in range(num_batches):
        if i % 10 == 0:
            print("Processed batch {} in {} seconds".format(i, time.time() - start))
            start = time.time()
        result = ged.compare(generated_graph + train_graph[i * batch_size: (i + 1) * batch_size],
                             [0])  # compare the batch to the generated molecule graph

        closest_molecules = add_molecules(closest_molecules, train, result)


    # Process the last batch if it didn't fit with the batch size
    if last_batch_size > 0:
        result = ged.compare(generated_graph + train_graph[-last_batch_size:], [0])
        closest_molecules = add_molecules(closest_molecules, train, result)
    # return the sorted molecules according to the edit distance
    return sorted(closest_molecules, key=lambda elem: elem[0], reverse=False)[:n]


if __name__ == "__main__":
    limit = 1
    train_set = utils.load_smiles_data("../data/train.txt", limit)
    #
    generated_beta0_001 = utils.load_smiles_data("../data/samples_MPNVAE-h450-L56-d3-beta0.001.txt", limit)
    generated_beta0_005 = utils.load_smiles_data("../data/samples_MPNVAE-h450-L56-d3-beta0.005.txt", limit)
    generated_noKL = utils.load_smiles_data("../data/samples_MPNVAE-h450-L56-d3-noKL.txt", limit)
    #
    # generated_molecule = generated_beta0_001[0]
    #
    # closest_molecules = get_closest_molecules_ged(1000, train_set, generated_molecule, 10)
    # print(closest_molecules)
    #
    # mols = [jtnn.chemutils.get_mol(generated_beta0_001[0])]
    # for _, smiles in closest_molecules:
    #     mols.append(jtnn.chemutils.get_mol(smiles))
    #
    # with open('../data/ged.txt', 'w') as f:
    #     f.write("{0:.2f} {1} \n".format(0, generated_beta0_001[0]))
    #     for mol in closest_molecules:
    #         f.write("{0:.2f} {1}\n".format(mol[0], mol[1]))
    #
    # images = Chem.Draw.MolsToImage(mols)
    #
    # images.save('closest_molecules.png')
    # print("External diversity: {}".format(external_diversity(train_set, generated_beta0_001)))
    # print("External diversity: {}".format(external_diversity(train_set, generated_beta0_005)))
    # print("External diversity: {}".format(external_diversity(train_set, generated_noKL)))
    print("Internal diversity: {}".format(internal_diversity(generated_beta0_001)))
    print("Internal diversity: {}".format(internal_diversity(generated_beta0_005)))
    print("Internal diversity: {}".format(internal_diversity(generated_noKL)))


