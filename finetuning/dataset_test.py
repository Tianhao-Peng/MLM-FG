import os
import csv
import math
import time
import random
import numpy as np
import torch
import collections
import sys
import networkx as nx
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from transformers import BertTokenizer

def rdchem_enum_to_dict(values):
    """values = {0: rdkit.Chem.rdchem.ChiralType.CHI_UNSPECIFIED, 
            1: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, 
            2: rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, 
            3: rdkit.Chem.rdchem.ChiralType.CHI_OTHER}
    """
    result = {}
    for i in range(len(values)):
        result[values[i]] = len(result)
    return result

bond_vocab_dict = {
    "bond_dir": rdchem_enum_to_dict(Chem.BondDir.values),
    "bond_type": rdchem_enum_to_dict(Chem.BondType.values),}

class MolTestDataset(Dataset):
    def __init__(self, dataset, labels, data_path, target, char2id, graph_char2id, regex, graph_padding, add_bos=True, add_eos=True):
        super(Dataset, self).__init__()
        # char2id['<cls>'] = len(char2id)
        # graph_char2id['<cls>'] = len(graph_char2id)
        
        self.dataset = dataset
        self.labels = labels
        self.char2id = char2id
        self.graph_char2id = graph_char2id
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.regex = regex
        self.graph_padding = graph_padding
        self.pad  = self.char2id['<pad>']
        self.mask = self.char2id['<mask>']
        self.eos  = self.char2id['<eos>']
        self.bos  = self.char2id['<bos>']
        self.cls  = self.char2id['<cls>']
        
        self.graph_pad  = self.graph_char2id['<pad>']
        self.graph_mask = self.graph_char2id['<mask>']
        self.graph_eos  = self.graph_char2id['<eos>']
        self.graph_bos  = self.graph_char2id['<bos>']

        self.conversion = 1
        if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')

    def __getitem__(self, index):
        tokens = self.dataset[index]
        label = self.labels[index] * self.conversion
        
        smiles_array = self.encode(tokens) # tokens to ids
        if self.graph_padding == 'False':
            graph_array, spd, adj, edge_index, edge_attr = self.smiles_to_2dgraph(tokens)
            hypergraph_adj = self.smiles_to_hypergraph(tokens)
            graph2smiles_pos = self.smiles_to_crossAdj(tokens, spd)
        elif self.graph_padding == 'True':
            graph_array, spd, adj, edge_index, edge_attr = self.smiles_to_2dgraph_padded(tokens)
            hypergraph_adj = self.smiles_to_hypergraph(tokens)
            graph2smiles_pos = torch.zeros((1,1)) #self.smiles_to_crossAdj(tokens, spd)
        
        return smiles_array, graph_array, adj, graph2smiles_pos, hypergraph_adj, edge_index, edge_attr, label

    def __len__(self):
        return len(self.dataset)
    
    def encode(self, smiles):
        if '\n' in smiles:
            print('carriage return in mol')
        char = self.regex.findall(smiles.strip('\n'))
        if self.add_bos == True:
            char = ['<bos>'] + char
        if self.add_eos == True:
            char = char + ['<eos>']
            
        tmp = []
        for word in char:
            if word not in self.char2id.keys():
                tmp.append(self.char2id['<unk>'])
            else:
                tmp.append(self.char2id[word])
        return torch.tensor(tmp)
        # return torch.tensor([self.char2id[word] for word in char])
    
    def shortest_path_length(self, graph):
        node_num = graph.number_of_nodes()
        sp_length = np.zeros([node_num, node_num])
        for node1, value in nx.shortest_path_length(graph):
            for node2, length in value.items():
                sp_length[node1][node2] = length
        return sp_length
    
    def adj_matrix(self, graph):
        node_num = graph.number_of_nodes()
        # adj_matrix = np.zeros([node_num, node_num])
        adj = nx.adj_matrix(graph).toarray()
        # print('adj', adj, type(adj))
        # for node_pair in nx.adj_matrix(graph):
        #     print('node_pair', node_pair)
        #     node1, node2 = node_pair
        #     adj_matrix[node1][node2] = 1
        # print(adj)
        return adj
        
        
    def smiles_to_2dgraph_padded(self, smiles):
        smiles_tokens = self.regex.findall(smiles.strip('\n'))
        smiles = "".join(smiles_tokens)
        mol = Chem.MolFromSmiles(smiles)
        graph = nx.Graph()
        node_list = []
        smiles_graph_map, graph_smiles_map = self.get_one_smiles_graph_map(smiles_tokens)
        graph_tokens = [self.graph_char2id['<pad>'] for i in range(len(smiles_tokens))]
        
        if self.add_bos == True:
            boffset = 1
            node_list = [self.graph_char2id['<bos>']] + node_list
            graph_tokens = [self.graph_char2id['<bos>']] + graph_tokens
            graph.add_node(0)
        else:
            boffset = 0

        for i in range(len(smiles_tokens)):
            graph.add_node(i+boffset)
        for atom in mol.GetAtoms():
            node_list.append(self.graph_char2id[atom.GetSymbol()])
            
        if self.add_eos == True:
            node_list = node_list + [self.graph_char2id['<eos>']]
            graph_tokens = graph_tokens + [self.graph_char2id['<eos>']]
            graph.add_node(graph.number_of_nodes())
            
        for i in range(len(graph_smiles_map)):
            graph_tokens[graph_smiles_map[i]] = node_list[i]
            
        edge_index, edge_attr = [[], []], []
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx() + boffset
            end_idx = bond.GetEndAtomIdx() + boffset
            graph.add_edge(graph_smiles_map[start_idx], graph_smiles_map[end_idx])
            edge_index[0].append(start_idx)
            edge_index[1].append(end_idx)
            bond_dir = bond.GetBondDir()
            bond_dir_id = bond_vocab_dict['bond_dir'][bond_dir] + 2 # 0: self-loops; 1: masking
            bond_type = bond.GetBondType()
            bond_type_id = bond_vocab_dict['bond_type'][bond_type] + 2 # 0: self-loops; 1: masking
            edge_attr.append([bond_dir_id, bond_type_id])
            
        pos = self.shortest_path_length(graph)
        adj = self.adj_matrix(graph)
        
        return torch.tensor(graph_tokens), torch.tensor(pos), torch.tensor(adj), torch.tensor(edge_index), torch.tensor(edge_attr)
    
    def smiles_to_2dgraph(self, smiles_tokens):
        smiles = "".join(smiles_tokens)
        node_list = []
        mol = Chem.MolFromSmiles(smiles)
        graph = nx.Graph()

        if self.add_bos == True:
            boffset = 1
            node_list = [self.graph_char2id['<bos>']] + node_list
            graph.add_node(0)
        else:
            boffset = 0
            
        for atom in mol.GetAtoms():
            node_list.append(self.graph_char2id[atom.GetSymbol()])
            graph.add_node(atom.GetIdx()+boffset, type=atom.GetAtomicNum())
            
        edge_index, edge_attr = [[], []], []
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx() + boffset
            end_idx = bond.GetEndAtomIdx() + boffset
            edge_index[0].append(start_idx)
            edge_index[1].append(end_idx)
            graph.add_edge(start_idx, end_idx, bond_type=bond.GetBondTypeAsDouble())
            bond_dir = bond.GetBondDir()
            bond_dir_id = bond_vocab_dict['bond_dir'][bond_dir] + 2 # 0: self-loops; 1: masking
            bond_type = bond.GetBondType()
            bond_type_id = bond_vocab_dict['bond_type'][bond_type] + 2 # 0: self-loops; 1: masking
            edge_attr.append([bond_dir_id, bond_type_id])

        if self.add_eos == True:
            node_list = node_list + [self.graph_char2id['<eos>']]
            graph.add_node(graph.number_of_nodes())

        pos = self.shortest_path_length(graph)
        adj = self.adj_matrix(graph)
        # print(pos)
        
        return torch.tensor(node_list), torch.tensor(pos), torch.tensor(adj), torch.tensor(edge_index), torch.tensor(edge_attr)


    def get_one_smiles_graph_map(self, smiles_tokens):
        if self.add_bos:
            boffset = 1
        else:
            boffset = 0
        # ['C', 'C', 'O', 'c', '1', 'c', 'c', '(', 'C', '=', 'N', 'N', 'C', '(', '=', 'O', ')', ...]
        smiles_graph_map = collections.defaultdict(list)
        graph_smiles_map = []#collections.defaultdict(list)
        if self.add_bos:
            smiles_graph_map[self.bos] = self.graph_bos # bos=0
            graph_smiles_map.append(self.bos) # bos=0
        # assert self.eos==self.graph_eos
        smiles = "".join(smiles_tokens)
        mol = Chem.MolFromSmiles(smiles)
        sidx = 0
        for i, atom in enumerate(mol.GetAtoms()):
            # node_list.append(atom.GetIdx())
            atom_symbol = atom.GetSymbol()
            # if charge > 0:
            #     atom_rec = '[{atom_rec}{charge}]'.format(atom_rec=atom_rec, charge='+')
            # elif charge < 0:
            #     atom_rec = '[{atom_rec}{charge}]'.format(atom_rec=atom_rec, charge='-')
            # else:
            #     atom_rec = atom_symbol
            while sidx < len(smiles_tokens):
                # transfrom SMILES to atom, match atom with graph's node
                smiles_atom = smiles_tokens[sidx].strip('[]').rstrip('+-0123456789').replace('@', '')
                if smiles_atom == 'H' and atom_symbol == 'H':
                    smiles_graph_map[sidx+boffset] = i+boffset
                    graph_smiles_map.append(sidx+boffset)
                    sidx += 1
                    break
                if smiles_atom.endswith('H') and atom_symbol != 'He':
                    smiles_atom = smiles_atom[:-1]
                if smiles_atom.upper() == atom_symbol.upper():
                    smiles_graph_map[sidx+boffset] = i+boffset
                    graph_smiles_map.append(sidx+boffset)
                    sidx += 1
                    break
                else:
                    sidx += 1
        if self.add_eos:
            eoffset = 1
            smiles_graph_map[len(smiles_tokens)+boffset] = mol.GetNumAtoms()+boffset
            graph_smiles_map.append(len(smiles_tokens)+boffset)
        else:
            eoffset = 0
        try:
            assert mol.GetNumAtoms()==(len(smiles_graph_map)-int(self.add_eos)-int(self.add_bos)), 'Error assert:mol.GetNumAtoms():{a}, len(smiles_graph_map):{length}, smiles_tokens:{smiles_tokens}, smiles_graph_map:{smiles_graph_map}, graph:{graph}'.format(a=mol.GetNumAtoms(), length=(len(smiles_graph_map)-int(self.add_eos)-int(self.add_bos)), smiles_tokens=smiles_tokens, smiles_graph_map=smiles_graph_map, graph=[atom.GetSymbol() for atom in mol.GetAtoms()])
            assert len(graph_smiles_map) == mol.GetNumAtoms()+boffset+eoffset, print(len(graph_smiles_map), mol.GetNumAtoms()+boffset+eoffset)
        except:
            tmp = [atom.GetSymbol() for atom in mol.GetAtoms()]
            for k, v in smiles_graph_map.items():
                print(smiles_tokens[k-boffset], tmp[v-boffset])
            sys.exit()

        return smiles_graph_map, graph_smiles_map
    
    def smiles_to_crossAdj(self, smiles, graph_pos):
        smiles_tokens = self.regex.findall(smiles.strip('\n'))
        smiles_graph_map, graph_smiles_map = self.get_one_smiles_graph_map(smiles_tokens)
        # smiles2graph_pos = torch.tensor((len(smiles_graph_map), len(graph_smiles_map)))
        if self.add_bos:
            boffset = 1
        else:
            boffset = 0
        if self.add_eos:
            eoffset = 1
        else:
            eoffset = 0
        graph2smiles_pos = torch.zeros((len(graph_smiles_map), len(smiles_tokens)+boffset+eoffset), dtype=torch.float64)
        graph2smiles_pos[:, graph_smiles_map] = graph_pos
        return graph2smiles_pos
        
    def smiles_to_hypergraph(self, smiles_tokens):
        # Define common functional groups using SMARTS
        functional_groups = ['[#6]-[#8]-[#6]', '[#6](=[#8])-[#8]-[#6]', '[#6](=[#8])-[#1]', '[#6](=[#8])-[#6]', '[#6](=[#8])-[#8][#1]', '[#6](=[#8])-[#7]', '[#6]-[#8][#1]', '[#6]-[#7]', '[#6]-[F,Cl,Br,I]', '*C(C)=O', '*=O']

        if self.add_eos:
            eoffset = 1
        else:
            eoffset = 0
           
        if self.add_bos:
            boffset = 1
        else:
            boffset = 0
        
        smiles = "".join(smiles_tokens)
        mol = Chem.MolFromSmiles(smiles)

        # Identify rings
        ring_info = mol.GetRingInfo()
        rings = ring_info.AtomRings()

        # Create hypergraph
        hyperedge_to_node = collections.defaultdict(list)
        node_to_hyperedge = collections.defaultdict(list)

        # Add rings to hypergraph
        for idx, ring in enumerate(rings):
            for atom in ring:
                hyperedge_to_node[f"ring_{idx}"].append(atom+boffset)
                node_to_hyperedge[atom+boffset].append(f"ring_{idx}")

        # Add functional groups to hypergraph
        for smarts in functional_groups:
            pattern = Chem.MolFromSmarts(smarts)
            matches = mol.GetSubstructMatches(pattern)
            for idx, match in enumerate(matches):
                for atom in match:
                    hyperedge_to_node[f"{smarts}_{idx}"].append(atom+boffset)
                    node_to_hyperedge[atom+boffset].append(f"{smarts}_{idx}")

        # Convert hypergraph to matrix
        num_atoms = mol.GetNumAtoms()
        num_hyperedges = len(hyperedge_to_node)
        matrix = np.zeros((num_atoms+eoffset+eoffset, num_hyperedges), dtype=int)
        # print('num_hyperedges', num_hyperedges)

        for idx, (key, atoms) in enumerate(hyperedge_to_node.items()):
            for atom in atoms:
                matrix[atom][idx] = 1
        if len(matrix[0]) > 40:
            matrix = matrix[:, :40]
        return torch.tensor(matrix, dtype=torch.int32)
    
    def get_mask(self, arrays):
        mask = (arrays==self.pad) #| (arrays==self.mask) | (arrays==self.eos) | (arrays==self.bos)
        return ~mask

    def my_collate_fn(self, batch):
        smiles_arrays, graph_arrays, adj_arrays, twod_position_embeddings, hypergraph_adjs, edge_index_l, edge_attr_l, labels = zip(*batch)
        
        max_smiles_len = max([s.size(0) for s in smiles_arrays])
        max_graph_len = max([g.size(0) for g in graph_arrays])
        max_twod_pos_emb_dim1 = max([t.size(0) for t in twod_position_embeddings])
        max_twod_pos_emb_dim2 = max([t.size(1) for t in twod_position_embeddings])
        max_adj_len = max([t.size(0) for t in adj_arrays])
        max_hypergraph_adj_dim1 = max([h.size(0) for h in hypergraph_adjs])
        max_hypergraph_adj_dim2 = max([h.size(1) for h in hypergraph_adjs])

        smiles_arrays_padded = torch.full((len(batch), max_smiles_len), self.pad)
        graph_arrays_padded = torch.full((len(batch), max_graph_len), self.pad)
        twod_position_embeddings_padded = torch.full((len(batch), max_twod_pos_emb_dim1, max_twod_pos_emb_dim2), 0)
        adj_position_embeddings_padded = torch.full((len(batch), max_adj_len, max_adj_len), 0)
        hypergraph_adjs_padded = torch.full((len(batch), max_hypergraph_adj_dim1, max_hypergraph_adj_dim2), 0)
        labels_tensor = torch.tensor(labels)

        g_node_num = len(graph_arrays[0])
        edge_index_merged = []
        for i, (s, g, a, t, h, _, _, _) in enumerate(batch):
            smiles_arrays_padded[i, :s.size(0)] = s
            graph_arrays_padded[i, :g.size(0)] = g
            adj_position_embeddings_padded[i, :a.size(0), :a.size(1)] = a
            twod_position_embeddings_padded[i, :t.size(0), :t.size(1)] = t
            hypergraph_adjs_padded[i, :h.size(0), :h.size(1)] = h
            edge_index_merged.append(edge_index_l[i]+g_node_num*i)
        edge_index_batch = torch.cat(edge_index_merged, dim=1).long()
        edge_attr_batch = torch.cat(edge_attr_l, dim=0).int()
        
        smiles_mask = self.get_mask(smiles_arrays_padded)
        graph_mask = self.get_mask(graph_arrays_padded)
        
        # print('smiles_arrays_padded', smiles_arrays_padded)
        # print('smiles_mask', smiles_mask)
        # print('graph_arrays_padded', graph_arrays_padded)
        # adj_position_embeddings_padded = adj_arrays
        
        # print('smiles_arrays_padded', smiles_arrays_padded)
        # print('smiles_mask', smiles_mask)

        return (smiles_arrays_padded, graph_arrays_padded, adj_position_embeddings_padded, twod_position_embeddings_padded, hypergraph_adjs_padded, 
                edge_index_batch, edge_attr_batch, labels_tensor, smiles_mask, graph_mask)

class MolTranBertTokenizer(BertTokenizer):
    def __init__(self, vocab_file: str = '',
                 regex_tokenizer=None,
                 do_lower_case=False,
                 unk_token='<pad>',
                 sep_token='<eos>',
                 pad_token='<pad>',
                 cls_token='<bos>',
                 mask_token='<mask>',
                 **kwargs):
        super().__init__(vocab_file,
                         unk_token=unk_token,
                         sep_token=sep_token,
                         pad_token=pad_token,
                         cls_token=cls_token,
                         mask_token=mask_token,
                         **kwargs)

        self.regex_tokenizer = regex_tokenizer
        self.wordpiece_tokenizer = None
        self.basic_tokenizer = None

    def _tokenize(self, text):
        split_tokens = self.regex_tokenizer.findall(text)
        return split_tokens

    def convert_tokens_to_string(self, tokens):
        out_string = "".join(tokens).strip()
        return out_string

class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labels, data_path, target, tokenizer):
        self.dataset = dataset
        self.labels = labels
        self.tokenizer = tokenizer
        
        self.conversion = 1
        if 'qm9' in data_path and target in ['homo', 'lumo', 'gap', 'zpve', 'u0']:
            self.conversion = 27.211386246
            print(target, 'Unit conversion needed!')

    def __getitem__(self, index):
        smiles = self.dataset[index]
        label = self.labels[index] * self.conversion
        
        return smiles, label
    
    def __len__(self):
        return len(self.dataset)
    
    def my_collate_fn(self, batch):
        tokens = self.tokenizer.batch_encode_plus([ smile[0] for smile in batch], padding=True, add_special_tokens=True)
        print((torch.tensor(tokens['input_ids'])[0], torch.tensor(tokens['attention_mask'])[0], torch.tensor([smile[1] for smile in batch])))
        return (torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask']), torch.tensor([smile[1] for smile in batch]))
        # return (smiles_arrays_padded, graph_arrays_padded, twod_position_embeddings_padded, hypergraph_adjs_padded, labels_tensor, smiles_mask, graph_mask)
