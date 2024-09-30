import socket
import glob
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
import networkx as nx
import torch
import numpy as np
import os
import shutil

def getipaddress():
    return socket.gethostbyname(socket.getfqdn())


def debug():
    print("Waiting for debugger to connect")
    if (
        socket.getfqdn().startswith("dcc")
        or socket.getfqdn().startswith("mol")
        or socket.getfqdn().startswith("ccc")
    ):
        debugpy.listen(address=(getipaddress(), 3000))
        debugpy.wait_for_client()
    debugpy.breakpoint()


class ListDataset:
    def __init__(self, seqs):
        self.seqs = seqs

    def __getitem__(self, index):
        return self.seqs[index]

    def __len__(self):
        return len(self.seqs)


def transform_single_embedding_to_multiple(smiles_z_map):
    """Transforms an embedding map of the format smi->embedding to
    smi-> {"canonical_embeddings":embedding}. This function exists
    as a compatibility layer

    Args:
        smiles_z_map ([type]): [description]
    """
    retval = dict()
    for key in smiles_z_map:
        retval[key] = {"canonical_embeddings": smiles_z_map[key]}
    return retval


def normalize_smiles(smi, canonical, isomeric):
    normalized = Chem.MolToSmiles(
        Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
    )
    return normalized


def get_all_proteins(affinity_dir: str):
    files = glob.glob(affinity_dir + "/*.csv")
    all_proteins = []
    print(files)
    for file in files:
        df = pd.read_csv(file)
        all_proteins.extend(df["protein"].tolist())
    return set(all_proteins)


def append_to_file(filename, line):
    with open(filename, "a") as f:
        f.write(line + "\n")


def write_to_file(filename, line):
    with open(filename, "w") as f:
        f.write(line + "\n")


def smiles2nxgraph(batch_smiles):
    '''
    batch_smiles: a batch of SMILES
    '''
    batch_graph = []
    for smiles_tokens in batch_smiles:
        smiles = "".join(smiles_tokens)
        print('smiles2graph', smiles)
        graph = nx.Graph()
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            graph.add_node(atom.GetIdx(), type=atom.GetAtomicNum())
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            graph.add_edge(start_idx, end_idx, bond_type=bond.GetBondTypeAsDouble())
        batch_graph.append(graph)
    return batch_graph

def smiles2graph(batch_smiles):
    batch_graph = [[], []]
    for smiles_tokens in batch_smiles:
        smiles = "".join(smiles_tokens)
        # print('smiles2graph', smiles)
        node_list, edge_index = [], [[], [], []]
        mol = Chem.MolFromSmiles(smiles)
        for atom in mol.GetAtoms():
            node_list.append(atom.GetAtomicNum())
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            edge_index[0].append(start_idx)
            edge_index[1].append(end_idx)
            edge_index[2].append(bond.GetBondTypeAsDouble())
        batch_graph[0].append(torch.tensor(node_list))
        batch_graph[1].append(torch.tensor(edge_index))
    return batch_graph

def DISCARD_smiles_to_2dgraph(batch_smiles):
    batch_graph = []
    pos_list = []
    max_node_num = 0
    for smiles_tokens in batch_smiles:
        smiles = "".join(smiles_tokens)
        node_list = []
        mol = Chem.MolFromSmiles(smiles)
        max_node_num = max(max_node_num, mol.GetNumAtoms())
        
    for smiles_tokens in batch_smiles:
        smiles = "".join(smiles_tokens)
        node_list = []
        mol = Chem.MolFromSmiles(smiles)
        graph = nx.Graph()
        for atom in mol.GetAtoms():
            node_list.append(atom.GetAtomicNum())
            graph.add_node(atom.GetIdx(), type=atom.GetAtomicNum())
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            graph.add_edge(start_idx, end_idx, bond_type=bond.GetBondTypeAsDouble())
            
        batch_graph.append(torch.tensor(node_list))
        pos_list.append(shortest_path_length(graph, max_node_num))
        # batch_graph[1].append(random_walk_probability_matrix(graph))
    
    pos_list = torch.from_numpy(np.array(pos_list))
    return batch_graph, pos_list

def smiles_to_3dgraph(batch_smiles):
    batch_graph = [[], []]
    for smiles_tokens in batch_smiles:
        smiles = "".join(smiles_tokens)
        # print('smiles2graph', smiles)
        node_list = []
        mol = Chem.MolFromSmiles(smiles)
        # graph = nx.Graph()
        for atom in mol.GetAtoms():
            node_list.append(atom.GetAtomicNum())
            
        batch_graph[0].append(torch.tensor(node_list))
        batch_graph[1].append(torch.tensor(spatial_distance_between_nodes(mol)))
    return batch_graph
    

def spatial_distance_between_nodes(mol):
    mol = Chem.RemoveHs(mol)  # Remove hydrogen atoms if desired
    
    # Generate 3D coordinate representations for each molecule
    # Get the atomic coordinates for the molecule
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    coords = mol.GetConformer().GetPositions()

    # Assign the coordinates to the tensor
    coords_tensor = torch.tensor(coords)
    # Reshape the tensor for efficient computation
    coords_tensor = coords_tensor.unsqueeze(1)  # Add a singleton dimension for broadcasting

    # Calculate pairwise distances using torch
    diff = coords_tensor - coords_tensor.transpose(1, 0)
    distances = torch.norm(diff, dim=-1)

    return distances

def shortest_path_length(graph, max_node_num):
    # node_num = graph.number_of_nodes()
    sp_length = np.ones([max_node_num, max_node_num])*-1
    for node1, value in nx.shortest_path_length(graph):
        for node2, length in value.items():
            sp_length[node1][node2] = length
    # sp_length = torch.nn.functional.normalize(sp_length, dim=None)
    return sp_length

def random_walk_probability_matrix(graph):
    adjacency_matrix = nx.to_numpy_matrix(graph)
    row_sums = np.sum(adjacency_matrix, axis=1).reshape(-1,1)
    # Normalize the adjacency matrix to obtain transition probabilities
    transition_matrix = adjacency_matrix / row_sums
    transition_matrix = torch.from_numpy(transition_matrix)
    return transition_matrix

    
def decompose2fragments(batch_smiles, batch_graph):
    '''
    batch_smiles: a batch of SMILES
    '''
    for smiles in batch_smiles:
        mol = Chem.MolFromSmiles(smiles)
        broken_bonds = BRICS.FindBRICSBonds(mol)
        while True:
            edge = next(broken_bonds)[0]
            
            
def graph_construction_and_featurization(smiles):
    """Construct graphs from SMILES and featurize them
    Parameters
    ----------
    smiles : list of str
        SMILES of molecules for embedding computation
    Returns
    -------
    list of DGLGraph
        List of graphs constructed and featurized
    list of bool
        Indicators for whether the SMILES string can be
        parsed by RDKit
    """
    graphs = []
    success = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                success.append(False)
                continue
            g = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=PretrainAtomFeaturizer(),
                               edge_featurizer=PretrainBondFeaturizer(),
                               canonical_atom_order=False)
            graphs.append(g)
            success.append(True)
        except:
            success.append(False)

    return graphs, success


def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            if 'lightning_logs' in home or 'checkpoints' in home:
                continue
            if (filename.endswith('.py') or filename.endswith('.sh')):
                Filelist.append(os.path.join(home, filename))
 
    return Filelist


def save_scripts(trainer, scripts=None):
    scripts_dir = os.path.join(trainer.logger.log_dir, 'scripts')
    os.mkdir(trainer.logger.log_dir)
    os.mkdir(scripts_dir)

    scripts_to_save = get_filelist('.')
    if scripts != None:
        scripts_to_save += scripts
    for script in scripts_to_save:
        dst_file = os.path.join(scripts_dir, os.path.basename(script))
        shutil.copyfile(script, dst_file)
    return 

def merge_one_matrix(hypergraph_adjs):
    device = hypergraph_adjs[0].device
    max_hypergraph_adj_dim1 = max([h.size(0) for h in hypergraph_adjs])
    max_hypergraph_adj_dim2 = max([h.size(1) for h in hypergraph_adjs])
    hypergraph_adjs_padded = torch.full((len(hypergraph_adjs), max_hypergraph_adj_dim1, max_hypergraph_adj_dim2), 0)

    for i, h in enumerate(hypergraph_adjs):
        hypergraph_adjs_padded[i, :h.size(0), :h.size(1)] = h
    return hypergraph_adjs_padded.to(device)