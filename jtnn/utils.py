


def load_smiles_data(data_path, limit=None):
    counter = 0
    data = []
    with open(data_path) as f:
        for line in f:
            s = line.split("\n")[0]
            data.append(s)
            counter += 1
            if limit is not None and counter >= limit:
                break

    return data