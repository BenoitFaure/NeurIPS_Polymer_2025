"""
datasets.py

This file contains dataset classes for the NeurIPS_Polymer_2025 project, including PolymerDataset.
These classes are designed for use with PyTorch Geometric and handle conversion of molecular data
from pandas DataFrames into graph-based formats.

PolymerDataset inputs:
    - root: Path to the directory where processed dataset files will be stored and loaded from.
    - df: pandas DataFrame containing at least a 'SMILES' column (molecular structure as string)
          and one or more label columns (properties or targets for each molecule).
"""

import torch
from torch_geometric.data import InMemoryDataset, Data

from rdkit import Chem
import pandas as pd

class PolymerDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset for polymer molecules.

    Converts a pandas DataFrame of SMILES strings and labels into a list of PyG Data objects,
    suitable for graph-based machine learning. Node features are extracted from atoms,
    edge indices from bonds, and labels/masks from DataFrame columns.
    """
    def __init__(self, root, df, transform=None, pre_transform=None):
        # Store the DataFrame containing SMILES and labels
        self.df = df
        # Initialize the parent InMemoryDataset
        super().__init__(root, transform, pre_transform)
        # Load processed data if available
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # No raw files needed; data is passed directly via DataFrame
        return []

    @property
    def processed_file_names(self):
        # Name of the processed data file
        return ['data.pt']

    def download(self):
        # No download step required; data is provided
        pass

    def process(self):
        """
        Processes the DataFrame into PyG Data objects.
        Each molecule is converted into a graph with node features, edge indices, labels, and masks.
        """
        data_list = []
        # Iterate through each row of the DataFrame
        for index, row in self.df.iterrows():
            smiles = row['SMILES']
            # Parse SMILES string into RDKit molecule
            mol = Chem.MolFromSmiles(smiles)
            
            # Skip molecules that RDKit cannot parse
            if mol is None:
                continue
            
            # --- Node Feature Construction (x) ---
            # For each atom, extract atomic number, degree, and implicit valence
            x = torch.tensor([[atom.GetAtomicNum(), atom.GetDegree(), atom.GetImplicitValence()]
                            for atom in mol.GetAtoms()], dtype=torch.float)
            
            # --- Edge Index Construction (edge_index) ---
            # For each bond, get atom indices and add both directions for undirected graph
            edge_indices = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices.extend([[i, j], [j, i]])
            # Convert edge list to tensor
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            
            # --- Label and Mask Construction (y and y_mask) ---
            # Extract label values (all columns except SMILES)
            labels = row[self.df.columns[1:]].values
            # Mask: True where label is not NaN
            y_mask = torch.tensor(~pd.isna(labels), dtype=torch.bool)
            # Replace NaNs with 0 for tensor compatibility
            y = torch.tensor(labels, dtype=torch.float)
            y[~y_mask] = 0.0
            
            # --- Create and append the Data object ---
            # Data object contains graph structure, features, labels, and mask
            data = Data(x=x, edge_index=edge_index, y=y.reshape(1, -1), y_mask=y_mask)
            data_list.append(data)

        # Apply optional pre-filtering
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        # Apply optional pre-transformations
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # Save the processed data for efficient future loading
        torch.save(self.collate(data_list), self.processed_paths[0])