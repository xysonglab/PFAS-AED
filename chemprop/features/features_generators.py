from typing import Callable, List, Union

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski

Molecule = Union[str, Chem.Mol]
FeaturesGenerator = Callable[[Molecule], np.ndarray]

FEATURES_GENERATOR_REGISTRY = {}


def register_features_generator(features_generator_name: str) -> Callable[[FeaturesGenerator], FeaturesGenerator]:
    """
    Registers a features generator.

    :param features_generator_name: The name to call the FeaturesGenerator.
    :return: A decorator which will add a FeaturesGenerator to the registry using the specified name.
    """

    def decorator(features_generator: FeaturesGenerator) -> FeaturesGenerator:
        FEATURES_GENERATOR_REGISTRY[features_generator_name] = features_generator
        return features_generator

    return decorator


def get_features_generator(features_generator_name: str) -> FeaturesGenerator:
    """
    Gets a registered FeaturesGenerator by name.

    :param features_generator_name: The name of the FeaturesGenerator.
    :return: The desired FeaturesGenerator.
    """
    if features_generator_name not in FEATURES_GENERATOR_REGISTRY:
        raise ValueError(f'Features generator "{features_generator_name}" could not be found. '
                         f'If this generator relies on rdkit features, you may need to install descriptastorus.')

    return FEATURES_GENERATOR_REGISTRY[features_generator_name]


def get_available_features_generators() -> List[str]:
    """Returns the names of available features generators."""
    return list(FEATURES_GENERATOR_REGISTRY.keys())


MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048


@register_features_generator('morgan')
def morgan_binary_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates a binary Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1-D numpy array containing the binary Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


@register_features_generator('morgan_count')
def morgan_counts_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates a counts-based Morgan fingerprint for a molecule.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing the counts-based Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    features_vec = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=num_bits)
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)

    return features


@register_features_generator('pfas')
def pfas_features_generator(mol: Molecule) -> np.ndarray:
    """
    Generates PFAS-specific features for a molecule.

    Features include:
    - Number of fluorine atoms
    - Number of C-F bonds
    - Length of perfluorinated carbon chain
    - Fluorination degree
    - Presence of specific PFAS functional groups
    - Carbon to fluorine ratio
    - Presence of ether linkages
    - Total carbon count
    - Presence of sulfonic/carboxylic acid groups

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :return: A 1D numpy array containing PFAS-specific features.
    """
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol

    if mol is None:
        return np.zeros(20)  # Return zero features if molecule is invalid

    features = []

    # 1. Number of fluorine atoms
    num_fluorine = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
    features.append(num_fluorine)

    # 2. Number of carbon atoms
    num_carbon = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    features.append(num_carbon)

    # 3. Number of C-F bonds
    num_cf_bonds = 0
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        if (atom1.GetAtomicNum() == 6 and atom2.GetAtomicNum() == 9) or \
                (atom1.GetAtomicNum() == 9 and atom2.GetAtomicNum() == 6):
            num_cf_bonds += 1
    features.append(num_cf_bonds)

    # 4. Fluorination degree (ratio of F to total heavy atoms)
    num_heavy_atoms = mol.GetNumHeavyAtoms()
    fluorination_degree = num_fluorine / num_heavy_atoms if num_heavy_atoms > 0 else 0
    features.append(fluorination_degree)

    # 5. Carbon to fluorine ratio
    cf_ratio = num_carbon / num_fluorine if num_fluorine > 0 else 0
    features.append(cf_ratio)

    # 6. Maximum continuous perfluorinated carbon chain length
    def get_perfluoro_chain_length(mol):
        """Find the longest chain of carbons where each carbon is bonded to fluorines."""
        max_length = 0
        visited = set()

        for atom_idx in range(mol.GetNumAtoms()):
            if atom_idx in visited:
                continue

            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() != 6:  # Not carbon
                continue

            # Check if this carbon is perfluorinated
            neighbors = [n.GetAtomicNum() for n in atom.GetNeighbors()]
            f_count = neighbors.count(9)

            if f_count >= 2:  # At least 2 fluorines (CF2 or CF3)
                # Start DFS to find chain length
                chain_length = 1
                stack = [(atom_idx, chain_length)]
                current_visited = {atom_idx}

                while stack:
                    curr_idx, curr_length = stack.pop()
                    max_length = max(max_length, curr_length)

                    curr_atom = mol.GetAtomWithIdx(curr_idx)
                    for neighbor in curr_atom.GetNeighbors():
                        n_idx = neighbor.GetIdx()
                        if n_idx not in current_visited and neighbor.GetAtomicNum() == 6:
                            # Check if neighbor carbon is also perfluorinated
                            n_neighbors = [n.GetAtomicNum() for n in neighbor.GetNeighbors()]
                            n_f_count = n_neighbors.count(9)
                            if n_f_count >= 1:  # At least 1 fluorine
                                current_visited.add(n_idx)
                                stack.append((n_idx, curr_length + 1))

                visited.update(current_visited)

        return max_length

    perfluoro_chain_length = get_perfluoro_chain_length(mol)
    features.append(perfluoro_chain_length)

    # 7. Number of CF3 groups
    cf3_pattern = Chem.MolFromSmarts('C(F)(F)F')
    num_cf3 = len(mol.GetSubstructMatches(cf3_pattern)) if cf3_pattern else 0
    features.append(num_cf3)

    # 8. Number of CF2 groups
    cf2_pattern = Chem.MolFromSmarts('[C;D4](F)(F)')
    num_cf2 = len(mol.GetSubstructMatches(cf2_pattern)) if cf2_pattern else 0
    features.append(num_cf2)

    # 9. Presence of sulfonic acid group (common in PFOS)
    sulfonic_pattern = Chem.MolFromSmarts('S(=O)(=O)O')
    has_sulfonic = 1 if sulfonic_pattern and mol.HasSubstructMatch(sulfonic_pattern) else 0
    features.append(has_sulfonic)

    # 10. Presence of carboxylic acid group (common in PFOA)
    carboxylic_pattern = Chem.MolFromSmarts('C(=O)O')
    has_carboxylic = 1 if carboxylic_pattern and mol.HasSubstructMatch(carboxylic_pattern) else 0
    features.append(has_carboxylic)

    # 11. Presence of ether linkages
    ether_pattern = Chem.MolFromSmarts('COC')
    num_ethers = len(mol.GetSubstructMatches(ether_pattern)) if ether_pattern else 0
    features.append(num_ethers)

    # 12. Molecular weight
    mol_weight = Descriptors.MolWt(mol)
    features.append(mol_weight / 100)  # Scale down

    # 13. LogP (hydrophobicity)
    logp = Descriptors.MolLogP(mol)
    features.append(logp)

    # 14. Number of rotatable bonds
    num_rotatable = Lipinski.NumRotatableBonds(mol)
    features.append(num_rotatable)

    # 15. Total degree of fluorination (F atoms per C atom)
    f_per_c = num_fluorine / num_carbon if num_carbon > 0 else 0
    features.append(f_per_c)

    # 16. Presence of branched perfluorinated groups
    branched_cf_pattern = Chem.MolFromSmarts('C(C(F)(F)F)(F)(F)')
    has_branched = 1 if branched_cf_pattern and mol.HasSubstructMatch(branched_cf_pattern) else 0
    features.append(has_branched)

    # 17. Number of oxygen atoms (relevant for PFAS degradation products)
    num_oxygen = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
    features.append(num_oxygen)

    # 18. Presence of phosphate groups (found in some PFAS)
    phosphate_pattern = Chem.MolFromSmarts('P(=O)(O)(O)O')
    has_phosphate = 1 if phosphate_pattern and mol.HasSubstructMatch(phosphate_pattern) else 0
    features.append(has_phosphate)

    # 19. Aromatic ring count
    num_aromatic_rings = Descriptors.NumAromaticRings(mol)
    features.append(num_aromatic_rings)

    # 20. Is it likely a PFAS compound (heuristic: has F and C-F bonds)
    is_likely_pfas = 1 if num_fluorine >= 3 and num_cf_bonds >= 3 else 0
    features.append(is_likely_pfas)

    return np.array(features, dtype=np.float32)


@register_features_generator('pfas_combined')
def pfas_combined_features_generator(mol: Molecule,
                                     radius: int = MORGAN_RADIUS,
                                     num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates combined Morgan fingerprint and PFAS-specific features.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing combined features.
    """
    # Get Morgan fingerprint
    morgan_features = morgan_binary_features_generator(mol, radius, num_bits)

    # Get PFAS features
    pfas_features = pfas_features_generator(mol)

    # Concatenate features
    combined_features = np.concatenate([morgan_features, pfas_features])

    return combined_features


@register_features_generator('pfas_morgan_counts')
def pfas_morgan_counts_features_generator(mol: Molecule,
                                          radius: int = MORGAN_RADIUS,
                                          num_bits: int = MORGAN_NUM_BITS) -> np.ndarray:
    """
    Generates combined counts-based Morgan fingerprint and PFAS-specific features.

    :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
    :param radius: Morgan fingerprint radius.
    :param num_bits: Number of bits in Morgan fingerprint.
    :return: A 1D numpy array containing combined features.
    """
    # Get counts-based Morgan fingerprint
    morgan_counts_features = morgan_counts_features_generator(mol, radius, num_bits)

    # Get PFAS features
    pfas_features = pfas_features_generator(mol)

    # Concatenate features
    combined_features = np.concatenate([morgan_counts_features, pfas_features])

    return combined_features


try:
    from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors


    @register_features_generator('rdkit_2d')
    def rdkit_2d_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D features for a molecule.

        :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdDescriptors.RDKit2D()
        features = generator.process(smiles)[1:]

        return features


    @register_features_generator('rdkit_2d_normalized')
    def rdkit_2d_normalized_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates RDKit 2D normalized features for a molecule.

        :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D normalized features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]

        return features


    @register_features_generator('rdkit_2d_pfas')
    def rdkit_2d_pfas_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates combined RDKit 2D features and PFAS-specific features.

        :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
        :return: A 1D numpy array containing combined features.
        """
        # Get RDKit 2D features
        rdkit_features = rdkit_2d_features_generator(mol)

        # Get PFAS features
        pfas_features = pfas_features_generator(mol)

        # Concatenate features
        combined_features = np.concatenate([rdkit_features, pfas_features])

        return combined_features


    @register_features_generator('rdkit_2d_normalized_pfas')
    def rdkit_2d_normalized_pfas_features_generator(mol: Molecule) -> np.ndarray:
        """
        Generates combined RDKit 2D normalized features and PFAS-specific features.

        :param mol: A molecule (i.e. either a SMILES string or an RDKit molecule).
        :return: A 1D numpy array containing combined features.
        """
        # Get RDKit 2D normalized features
        rdkit_features = rdkit_2d_normalized_features_generator(mol)

        # Get PFAS features
        pfas_features = pfas_features_generator(mol)

        # Concatenate features
        combined_features = np.concatenate([rdkit_features, pfas_features])

        return combined_features

except ImportError:
    pass

