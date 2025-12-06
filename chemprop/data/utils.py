from argparse import Namespace
import csv
import os
from logging import Logger
import pickle
import random
from typing import List, Set, Tuple
from collections import defaultdict
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm, trange
import torch

from .data import MoleculeDatapoint, MoleculeDataset
from .scaffold import log_scaffold_stats, scaffold_split
from chemprop.features import load_features


def get_task_names(path: str, use_compound_names: bool = False) -> List[str]:
    """
    Gets the task names from a data CSV file.

    :param path: Path to a CSV file.
    :param use_compound_names: Whether file has compound names in addition to smiles strings.
    :return: A list of task names.
    """
    index = 2 if use_compound_names else 1
    task_names = get_header(path)[index:]

    return task_names


def get_header(path: str) -> List[str]:
    """
    Returns the header of a data CSV file.

    :param path: Path to a CSV file.
    :return: A list of strings containing the strings in the comma-separated header.
    """
    with open(path) as f:
        header = next(csv.reader(f))

    return header


def get_num_tasks(path: str) -> int:
    """
    Gets the number of tasks in a data CSV file.

    :param path: Path to a CSV file.
    :return: The number of tasks.
    """
    return len(get_header(path)) - 1


def get_smiles(path: str) -> List[str]:
    """
    Returns the smiles strings from a data CSV file (assuming the first line is a header).

    :param path: Path to a CSV file
    :return: A list of smiles strings.
    """
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        smiles = [line[0] for line in reader]

    return smiles


def filter_invalid_smiles(data: MoleculeDataset) -> MoleculeDataset:
    """
    Filters out invalid SMILES.

    :param data: A MoleculeDataset.
    :return: A MoleculeDataset with only valid molecules.
    """
    return MoleculeDataset([datapoint for datapoint in data
                            if datapoint.smiles != '' and datapoint.mol is not None
                            and datapoint.mol.GetNumHeavyAtoms() > 0])


def get_data(path: str,
             skip_invalid_smiles: bool = True,
             args: Namespace = None,
             features_path: List[str] = None,
             max_data_size: int = None,
             use_compound_names: bool = False,
             logger: Logger = None) -> MoleculeDataset:
    debug = logger.debug if logger is not None else print

    if args is not None:
        max_data_size = max_data_size or float('inf')
        features_path = features_path or args.features_path
    else:
        max_data_size = max_data_size or float('inf')

    # Load features
    if features_path is not None:
        features_data = []
        for feat_path in features_path:
            features_data.append(load_features(feat_path))  # each is num_data x num_features
        features_data = np.concatenate(features_data, axis=1)
    else:
        features_data = None

    skip_smiles = set()

    # Load data
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        lines = []
        for line in reader:
            smiles = line[0]

            if smiles in skip_smiles:
                continue

            lines.append(line)

            if len(lines) >= max_data_size:
                break

        data = MoleculeDataset([
            MoleculeDatapoint(
                line=line,
                args=args,
                features=features_data[i] if features_data is not None else None,
                use_compound_names=use_compound_names
            ) for i, line in tqdm(enumerate(lines), total=len(lines))
        ])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    if data.data[0].features is not None:
        args.features_dim = len(data.data[0].features)

    return data


def get_data_from_smiles(smiles: List[str], skip_invalid_smiles: bool = True, logger: Logger = None) -> MoleculeDataset:

    debug = logger.debug if logger is not None else print

    data = MoleculeDataset([MoleculeDatapoint([smile]) for smile in smiles])

    # Filter out invalid SMILES
    if skip_invalid_smiles:
        original_data_len = len(data)
        data = filter_invalid_smiles(data)

        if len(data) < original_data_len:
            debug(f'Warning: {original_data_len - len(data)} SMILES are invalid.')

    return data


def split_data(data: MoleculeDataset,
               split_type: str = 'random',
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0,
               args: Namespace = None,
               logger: Logger = None) -> Tuple[MoleculeDataset,
MoleculeDataset,
MoleculeDataset]:

    assert len(sizes) == 3 and sum(sizes) <= 1

    if args is not None:
        folds_file, val_fold_index, test_fold_index = \
            args.folds_file, args.val_fold_index, args.test_fold_index
    else:
        folds_file = val_fold_index = test_fold_index = None

    if split_type == 'predetermined':
        if not val_fold_index:
            assert sizes[2] == 0  # test set is created separately so use all of the other data for train and val
        assert folds_file is not None
        assert test_fold_index is not None

        try:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f)
        except UnicodeDecodeError:
            with open(folds_file, 'rb') as f:
                all_fold_indices = pickle.load(f, encoding='latin1')  # in case we're loading indices from python2
        assert len(data) == sum([len(fold_indices) for fold_indices in all_fold_indices])

        log_scaffold_stats(data, all_fold_indices, logger=logger)

        folds = [[data[i] for i in fold_indices] for fold_indices in all_fold_indices]

        test = folds[test_fold_index]
        if val_fold_index is not None:
            val = folds[val_fold_index]

        train_val = []
        for i in range(len(folds)):
            if i != test_fold_index and (val_fold_index is None or i != val_fold_index):
                train_val.extend(folds[i])

        if val_fold_index is not None:
            train = train_val
        else:
            random.seed(seed)
            random.shuffle(train_val)
            train_size = int(sizes[0] * len(train_val))
            train = train_val[:train_size]
            val = train_val[train_size:]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    elif split_type == 'scaffold_balanced':
        return scaffold_split(data, sizes=sizes, balanced=True, seed=seed, logger=logger)

    elif split_type == 'scaffold':
        return scaffold_split(data, sizes=sizes, balanced=False, seed=seed, logger=logger)

    elif split_type == 'ood_test':

        os.makedirs(args.ood_save_dir, exist_ok=True)
        dataset_name = args.data_path.rsplit("/", 1)[-1].split(".")[0]
        save_file_name = os.path.join(args.ood_save_dir, f"{dataset_name}.p")

        def tanimoto_sim_mat(x, y):
            # Calculate tanimoto distance with binary fingerprint
            intersection_mat = x[:, None, :] & y[None, :, :]
            union_mat = x[:, None, :] | y[None, :, :]

            intersection = intersection_mat.sum(-1)
            union = union_mat.sum(-1)
            return intersection / union

        def tanimoto_in_chunks(x, y, max_second=40):
            """ Compute tanimoto_sim_mat but break y into chunks
            to avoid memory issue """
            # Return tanimoto similarity
            num_splits = (len(y) // max_second + 1)
            train_fp_chunks = np.array_split(y, num_splits)
            dist_list = []
            for x2 in tqdm(train_fp_chunks):
                sim_mat = tanimoto_sim_mat(x, x2)
                dist_list.append(sim_mat)
            sim_mat = np.hstack(dist_list)
            return sim_mat

            # Sort data for consistency

        data.sort(key=lambda x: x.smiles)

        # Create fingerprints
        mols = data.mols()
        fps = []
        fp_dict = {}
        for mol_entry in data:
            mol = mol_entry.mol
            smi = mol_entry.smiles
            arr = np.zeros((0,), dtype=np.int8)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            DataStructs.ConvertToNumpyArray(fp, arr)
            fps.append(fp)
            fp_dict[smi] = arr
        fps = np.vstack(fps)

        all_indices = np.arange(len(mols))
        current_train = np.ones(len(all_indices)).astype(bool)
        current_val = np.zeros(len(all_indices)).astype(bool)
        current_test_ood = np.zeros(len(all_indices)).astype(bool)
        current_test_id = np.zeros(len(all_indices)).astype(bool)
        current_test = np.zeros(len(all_indices)).astype(bool)

        # Compute all by all tani_sim
        print("Starting to compute all by all tani sim")

        # All by all tanimoto similarity computation
        if os.path.isfile(save_file_name):
            ## LOAD from file
            print("Loading tani sim from file")
            tani_sim = pickle.load(open(save_file_name, "rb"))
        else:
            ## SAVE into file
            print("Creating tani sim.")
            tani_sim = tanimoto_in_chunks(fps, fps, max_second=10)
            print("Writing tani sim into file")
            pickle.dump(tani_sim, open(save_file_name, "wb"))

        num_val = int(sizes[1] * len(data))
        val_indices = np.random.choice(a=all_indices, size=num_val,
                                       replace=False)

        current_val[val_indices] = True
        current_train[val_indices] = False

        amount_in_test = int(sizes[2] * len(data))
        amount_ood = int(0.5 * amount_in_test)

        modified_sim = tani_sim.copy()

        # Make self similarity = -1
        modified_sim[all_indices, all_indices] = -1

        modified_sim[:, val_indices] = -1
        modified_sim[val_indices, val_indices] = 1

        n_num = 10
        # print("Starting to add OOD values")
        for _ in tqdm(range(amount_ood)):
            top_n = np.partition(modified_sim, -n_num, axis=1)[:, -n_num:]
            top_n_mean = np.mean(top_n, 1)
            new_test_index = np.argmin(top_n_mean)

            modified_sim[:, new_test_index] = -1
            modified_sim[new_test_index, :] = 1

            current_test_ood[new_test_index] = True
            current_train[new_test_index] = False

        # Now choose in domain samples
        modified_sim[current_test_ood, :] = -1
        modified_sim[current_val, :] = -1
        print("Starting to add ID values")
        for _ in tqdm(range(amount_ood)):
            top_n = np.partition(modified_sim, -n_num, axis=1)[:, -n_num:]
            top_n_mean = np.mean(top_n, 1)

            new_test_index = np.argmax(top_n_mean)

            modified_sim[:, new_test_index] = -1
            modified_sim[new_test_index, :] = -1

            current_test_id[new_test_index] = True
            current_train[new_test_index] = False

        # Define one current test
        current_test[current_test_id] = True
        current_test[current_test_ood] = True

        # Subset the tani_sim matrix with test and train
        sim_mat = tani_sim[current_test, :][:, current_train]

        ## Get top_n max sim for each item in train
        n_num_eval = 10
        max_sims = np.mean(np.sort(sim_mat, 1)[:, ::-1][:, :n_num_eval], 1)

        # Now choose val samples
        train_indices = set(np.argwhere(current_train).flatten().tolist())
        test_indices = set(np.argwhere(current_test).flatten().tolist())
        val_indices = set(np.argwhere(current_val).flatten().tolist())

        train_indices = sorted(list(train_indices))
        val_indices = sorted(list(val_indices))
        test_indices = sorted(list(test_indices))

        res = []
        smi_to_ood = dict()
        for index, (test_index, test_max_sim) in enumerate(zip(test_indices,
                                                               max_sims)):
            smi = data[test_index].smiles
            partition = "ood" if current_test_ood[test_index] else "id"
            smi_to_ood[smi] = partition
            entry = {"smiles": smi,
                     "partition": partition,
                     "max_sim_to_train": test_max_sim}
            res.append(entry)

        df = pd.DataFrame(res)

        train = [data[i] for i in train_indices]
        val = [data[i] for i in val_indices]
        test = [data[i] for i in test_indices]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test), df

    elif split_type == 'random':
        data.shuffle(seed=seed)

        train_size = int(sizes[0] * len(data))
        train_val_size = int((sizes[0] + sizes[1]) * len(data))
        train_val_test_size = int((sizes[0] + sizes[1] + sizes[2]) * len(data))

        train = data[:train_size]

        #### If you want to balance Stokes primary data for debugging, do it here
        if args.stokes_balance != 1.0:
            train_balance = []

            for mol in train:
                if mol.targets[0] < 0.2:
                    train_balance.append(mol)
                else:
                    if np.random.rand() < args.stokes_balance:
                        train_balance.append(mol)
            train = train_balance

        val = data[train_size:train_val_size]
        test = data[train_val_size:train_val_test_size]

        return MoleculeDataset(train), MoleculeDataset(val), MoleculeDataset(test)

    else:
        raise ValueError(f'split_type "{split_type}" not supported.')


def validate_data(data_path: str) -> Set[str]:
    """
    Validates a data CSV file, returning a set of errors.

    :param data_path: Path to a data CSV file.
    :return: A set of error messages.
    """
    errors = set()

    header = get_header(data_path)

    with open(data_path) as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        smiles, targets = [], []
        for line in reader:
            smiles.append(line[0])
            targets.append(line[1:])

    # Validate header
    if len(header) == 0:
        errors.add('Empty header')
    elif len(header) < 2:
        errors.add('Header must include task names.')

    mol = Chem.MolFromSmiles(header[0])
    if mol is not None:
        errors.add('First row is a SMILES string instead of a header.')

    # Validate smiles
    for smile in tqdm(smiles, total=len(smiles)):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            errors.add('Data includes an invalid SMILES.')

    # Validate targets
    num_tasks_set = set(len(mol_targets) for mol_targets in targets)
    if len(num_tasks_set) != 1:
        errors.add('Inconsistent number of tasks for each molecule.')

    if len(num_tasks_set) == 1:
        num_tasks = num_tasks_set.pop()
        if num_tasks != len(header) - 1:
            errors.add('Number of tasks for each molecule doesn\'t match number of tasks in header.')

    unique_targets = set(np.unique([target for mol_targets in targets for target in mol_targets]))

    if unique_targets <= {''}:
        errors.add('All targets are missing.')

    for target in unique_targets - {''}:
        try:
            float(target)
        except ValueError:
            errors.add('Found a target which is not a number.')

    return errors


def get_data_batches(data: MoleculeDataset,
                     iter_size: int,
                     use_last: bool = False,
                     shuffle: bool = False,
                     quiet: bool = False):
    """
    Yield batch, features_batch, target_batch

    :param data: Data to be batched
    :param iter_size: Batch size
    :param use_last: If true, return the last batch as well
    :param shuffle: If true, shuffle the data
    :param quiet: If true, run on quiet
    :return: (batch, features, targets, batch_len)
    """
    if shuffle:
        data.shuffle()

    if not use_last:
        num_iters = len(data) // iter_size * iter_size  # don't use the last batch if it's small, for stability
    else:
        num_iters = len(data)

    my_range = range if quiet else trange

    for i in my_range(0, num_iters, iter_size):
        if i + iter_size > len(data) and not use_last:
            break

        mol_batch = data[i:i + iter_size]
        mol_batch = MoleculeDataset(mol_batch)
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = smiles_batch

        yield (batch, features_batch, target_batch, len(mol_batch))