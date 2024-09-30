import regex as re
import torch
import numpy as np
import random
import collections
import json
import networkx as nx
import os
from rdkit import Chem

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


FUNCTIONAL_GROUPS = ['[#6]-[#8]-[#6]', '[#6](=[#8])-[#8]-[#6]', '[#6](=[#8])-[#1]', '[#6](=[#8])-[#6]', '[#6](=[#8])-[#8][#1]', '[#6](=[#8])-[#7]', '[#6]-[#8][#1]', '[#6]-[#7]', '[#6]-[F,Cl,Br,I]', '*C(C)=O', '*=O']

FUNCTIONAL_GROUPS_ALL = [
        # C
        "[CX4]",
        "[$([CX2](=C)=C)]",
        "[$([CX3]=[CX3])]",
        "[$([CX2]#C)]",
        # C & O
        "[CX3]=[OX1]",
        "[$([CX3]=[OX1]),$([CX3+]-[OX1-])]",
        "[CX3](=[OX1])C",
        "[OX1]=CN",
        "[CX3](=[OX1])O",
        "[CX3](=[OX1])[F,Cl,Br,I]",
        "[CX3H1](=O)[#6]",
        "[CX3](=[OX1])[OX2][CX3](=[OX1])",
        "[NX3][CX3](=[OX1])[#6]",
        "[NX3][CX3]=[NX3+]",
        "[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]",
        "[NX3][CX3](=[OX1])[OX2H0]",
        "[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]",
        "[CX3](=O)[O-]",
        "[CX3](=[OX1])(O)O",
        "[CX3](=[OX1])([OX2])[OX2H,OX1H0-1]",
        "C[OX2][CX3](=[OX1])[OX2]C",
        "[CX3](=O)[OX2H1]",
        "[CX3](=O)[OX1H0-,OX2H1]",
        "[NX3][CX2]#[NX1]",
        "[#6][CX3](=O)[OX2H0][#6]",
        "[#6][CX3](=O)[#6]",
        "[OD2]([#6])[#6]",
        # H
        "[H]",
        "[!#1]",
        "[H+]",
        "[+H]",
        "[!H]",
        # N
        "[NX3;H2,H1;!$(NC=O)]",
        "[NX3][CX3]=[CX3]",
        "[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]",
        "[NX3;H2,H1;!$(NC=O)].[NX3;H2,H1;!$(NC=O)]",
        "[NX3][$(C=C),$(cc)]",
        "[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[O,N]",
        "[NX3H2,NH3X4+][CX4H]([*])[CX3](=[OX1])[NX3,NX4+][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-]",
        "[$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H]([*])[CX3](=[OX1])[OX2H,OX1-,N]",
        "[CH3X4]",
        "[CH2X4][CH2X4][CH2X4][NHX3][CH0X3](=[NH2X3+,NHX2+0])[NH2X3]",
        "[CH2X4][CX3](=[OX1])[NX3H2]",
        "[CH2X4][CX3](=[OX1])[OH0-,OH]",
        "[CH2X4][SX2H,SX1H0-]",
        "[CH2X4][CH2X4][CX3](=[OX1])[OH0-,OH]",
        "[$([$([NX3H2,NX4H3+]),$([NX3H](C)(C))][CX4H2][CX3](=[OX1])[OX2H,OX1-,N])]",
        "[CH2X4][#6X3]1:[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]:\
[$([#7X3H+,#7X2H0+0]:[#6X3H]:[#7X3H]),$([#7X3H])]:[#6X3H]1",
        "[CHX4]([CH3X4])[CH2X4][CH3X4]",
        "[CH2X4][CHX4]([CH3X4])[CH3X4]",
        "[CH2X4][CH2X4][CH2X4][CH2X4][NX4+,NX3+0]",
        "[CH2X4][CH2X4][SX2][CH3X4]",
        "[CH2X4][cX3]1[cX3H][cX3H][cX3H][cX3H][cX3H]1",
        "[$([NX3H,NX4H2+]),$([NX3](C)(C)(C))]1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[OX2H,OX1-,N]",
        "[CH2X4][OX2H]",
        "[NX3][CX3]=[SX1]",
        "[CHX4]([CH3X4])[OX2H]",
        "[CH2X4][cX3]1[cX3H][nX3H][cX3]2[cX3H][cX3H][cX3H][cX3H][cX3]12",
        "[CH2X4][cX3]1[cX3H][cX3H][cX3]([OHX2,OH0X1-])[cX3H][cX3H]1",
        "[CHX4]([CH3X4])[CH3X4]",
        "N[CX4H2][CX3](=[OX1])[O,N]",
        "N1[CX4H]([CH2][CH2][CH2]1)[CX3](=[OX1])[O,N]",
        "[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]",
        "[$([NX1-]=[NX2+]=[NX1-]),$([NX1]#[NX2+]-[NX1-2])]",
        "[#7]",
        "[NX2]=N",
        "[NX2]=[NX2]",
        "[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]",
        "[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]",
        "[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]",
        "[NX3][NX3]",
        "[NX3][NX2]=[*]",
        "[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]",
        "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
        "[NX3+]=[CX3]",
        "[CX3](=[OX1])[NX3H][CX3](=[OX1])",
        "[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])",
        "[CX3](=[OX1])[NX3H0]([NX3H0]([CX3](=[OX1]))[CX3](=[OX1]))[CX3](=[OX1])",
        "[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]",
        "[$([OX1]=[NX3](=[OX1])[OX1-]),$([OX1]=[NX3+]([OX1-])[OX1-])]",
        "[NX1]#[CX2]",
        "[CX1-]#[NX2+]",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8].[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]",
        "[NX2]=[OX1]",
        "[$([#7+][OX1-]),$([#7v5]=[OX1]);!$([#7](~[O])~[O]);!$([#7]=[#7])]",
        # O
        "[OX2H]",
        "[#6][OX2H]",
        "[OX2H][CX3]=[OX1]",
        "[OX2H]P",
        "[OX2H][#6X3]=[#6]",
        "[OX2H][cX3]:[c]",
        "[OX2H][$(C=C),$(cc)]",
        "[$([OH]-*=[!#6])]",
        "[OX2,OX1-][OX2,OX1-]",
        # P
        "[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),\
$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-])\
,$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]",
        "[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),\
$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),\
$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]",
        # S
        "[S-][CX3](=S)[#6]",
        "[#6X3](=[SX1])([!N])[!N]",
        "[SX2]",
        "[#16X2H]",
        "[#16!H0]",
        "[#16X2H0]",
        "[#16X2H0][!#16]",
        "[#16X2H0][#16X2H0]",
        "[#16X2H0][!#16].[#16X2H0][!#16]",
        "[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]",
        "[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]",
        "[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]",
        "[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]",
        "[SX4](C)(C)(=O)=N",
        "[$([SX4](=[OX1])(=[OX1])([!O])[NX3]),$([SX4+2]([OX1-])([OX1-])([!O])[NX3])]",
        "[$([#16X3]=[OX1]),$([#16X3+][OX1-])]",
        "[$([#16X3](=[OX1])([#6])[#6]),$([#16X3+]([OX1-])([#6])[#6])]",
        "[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]",
        "[$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)]",
        "[$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6]),$([#16X4](=[OX1])(=[OX1])([OX2][#6])[OX2][#6])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]",
        "[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]",
        "[#16X2][OX2H,OX1H0-]",
        "[#16X2][OX2H0]",
        # X
        "[#6][F,Cl,Br,I]",
        "[F,Cl,Br,I]",
        "[F,Cl,Br,I].[F,Cl,Br,I].[F,Cl,Br,I]",
    ]

class Encoder():

    def __init__(self, max_length=500, add_bos=True, add_eos=True, feature_size=32, batch_size=-1, vocab_path='', pretrain_style='', graph_padding=''):
        path = os.path.join('../data/pretrain/', vocab_path, 'extend-{}-smiles-vocab-full.json'.format(vocab_path))
        with open(path, 'r') as f:
            self.vocab_encoder = json.load(f)
            
        self.max_length = max_length
        self.min_length = 1
        self.mod_length = 42
        self.mlm_probability = .1
        self.avg_length = 66
        self.tail = 122
        self.b0_cache=collections.deque()
        self.b1_cache=collections.deque()
        self.b2_cache=collections.deque()
        self.b3_cache=collections.deque()
        self.bucket0=collections.deque()
        self.bucket1=collections.deque()
        self.bucket2=collections.deque()
        self.bucket3=collections.deque()
        
        self.graph_padding = graph_padding

        self.b0_max=1100
        self.b1_max=700
        self.b2_max=150
        self.b3_max=50

        values = list(self.vocab_encoder.values())
        num_top = 0
        middle_top = 0
        bottom = 0
        for  count in values:
            if count > 100000:
                num_top += 1
            if count > 50:
                middle_top += 1
        middle_top = middle_top - num_top
        self.cutoffs = [num_top+4, middle_top]
        self.char2id = {"<bos>":0, "<eos>":1, "<pad>":2, "<mask>":3}
        self.id2char = {0:"<bos>", 1:"<eos>", 2:"<pad>", 3:"<mask>"}
        self.pad  = self.char2id['<pad>']
        self.mask = self.char2id['<mask>']
        self.eos  = self.char2id['<eos>']
        self.bos  = self.char2id['<bos>']
        
        pos = 0
        for key, value in self.vocab_encoder.items():
            self.char2id[key] = pos+4
            self.id2char[pos+4] = key
            pos += 1
        self.char2id["<unk>"] = pos + 4
        self.id2char[pos+4] = "<unk>"
        pos += 1
        self.char2id["<cls>"] = pos + 4
        self.id2char[pos+4] = "<cls>"
        self.pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)
        self.add_bos = add_bos
        self.add_eos = add_eos
        
        self.graph_char2id = {"<bos>":0, "<eos>":1, "<pad>":2, "<mask>":3}
        self.graph_pad  = self.graph_char2id['<pad>']
        self.graph_mask = self.graph_char2id['<mask>']
        self.graph_eos  = self.graph_char2id['<eos>']
        self.graph_bos  = self.graph_char2id['<bos>']
        with open('./periodic_table_of_elements.json', 'r') as f:
            element_to_index = json.load(f) # 'H': 1, ...
            for key, value in element_to_index.items():
                self.graph_char2id[key] = value + 3
        self.graph_char2id["<unk>"] = len(self.graph_char2id)
        self.graph_char2id["<cls>"] = len(self.graph_char2id)
        # self.new_word = defaultdict(int)
        
        assert pretrain_style in ['subgraph', 'atom', 'subgraph_and_atom', 'subsequence']
        self.pretrain_style = pretrain_style
        
    def encode(self, char):
        #if len(char) > self.max_length:
        #    char = char[:self.max_length]
        if self.add_bos == True:
            char = ['<bos>'] + char
        if self.add_eos == True:
            char = char + ['<eos>']
        
        return torch.tensor([self.char2id[word] for word in char])

    def encoder(self, tokens):
        #return *map(lambda x: self.encode(x), tokens)
        return [self.encode(mol) for mol in tokens]

    def process_text(self, text):
        #random length sequences seems to help training
        mod_length = self.mod_length #+ random.randint(-1, 3)
        avg_length = self.avg_length #+ random.randint(-3, 5)
        for mol in text:
            #fill up buckets and caches
            if '\n' in mol['text']:
                print('carriage return in mol')
            raw_regex = self.regex.findall(mol['text'].strip('\n'))
            length = len(raw_regex)
            if length > self.min_length and length < mod_length:
                if len(self.bucket0) < self.b0_max:
                    self.bucket0.append(raw_regex)
                else:
                    self.b0_cache.append(raw_regex)
            elif length >= mod_length and length < avg_length:
                if len(self.bucket1) < self.b1_max:
                    self.bucket1.append(raw_regex)
                else:
                    self.b1_cache.append(raw_regex)
            elif length >= avg_length and length < self.tail:
                self.b2_cache.append(raw_regex)
            elif length >= self.tail and length < self.max_length:
                self.b3_cache.append(raw_regex)
        #pour cache elements into any open bucket
        if len(self.bucket0) < self.b0_max and len(self.b0_cache) > 0:
            cache_size = len(self.b0_cache)
            max_margin = self.b0_max-len(self.bucket0)
            range0 = min(cache_size, max_margin)
            outbucket0 = [self.bucket0.pop() for item in range(len(self.bucket0))] + [self.b0_cache.pop() for i in range(range0)]
        else:
            outbucket0 = [self.bucket0.pop() for item in range(len(self.bucket0))]
        if len(self.bucket1) < self.b1_max and len(self.b1_cache) > 0:
            cache_size = len(self.b1_cache)
            max_margin = self.b1_max-len(self.bucket1)
            range1 = min(cache_size, max_margin)
            outbucket1 = [self.bucket1.pop() for item in range(len(self.bucket1))] + [self.b1_cache.pop() for i in range(range1)]
        else:
            outbucket1 = [self.bucket1.pop() for item in range(len(self.bucket1))]

        if len(self.b2_cache) > self.b2_max:
            cache_size = len(self.b2_cache)
            max_margin = self.b2_max
            range2 = min(cache_size, max_margin)
            outbucket2 =  [self.b2_cache.pop() for i in range(range2)]
        else:
            outbucket2=[]
        if len(self.b3_cache) > self.b3_max:
            cache_size = len(self.b3_cache)
            max_margin = self.b3_max
            range3 = min(cache_size, max_margin)
            outbucket3 =  [self.b3_cache.pop() for i in range(range3)]
        else:
            outbucket3 = []
        return outbucket0, outbucket1, outbucket2, outbucket3
    
    def process_cache(self):
        #pour cache elements into any open bucket, when cache is full
        if len(self.b0_cache) > self.b0_max:
            cache_size = len(self.b0_cache)
            max_margin = self.b0_max
            range0 = min(cache_size, max_margin)
            outbucket0 =  [self.b0_cache.pop() for i in range(range0)]
        else:
            outbucket0=[]
        if len(self.b1_cache) > self.b1_max:
            cache_size = len(self.b1_cache)
            max_margin = self.b1_max
            range1 = min(cache_size, max_margin)
            outbucket1 =  [self.b1_cache.pop() for i in range(range1)]
        else:
            outbucket1=[]
        if len(self.b2_cache) > self.b2_max:
            cache_size = len(self.b2_cache)
            max_margin = self.b2_max
            range2 = min(cache_size, max_margin)
            outbucket2 =  [self.b2_cache.pop() for i in range(range2)]
        else:
            outbucket2=[]
        if len(self.b3_cache) > self.b3_max:
            cache_size = len(self.b3_cache)
            max_margin = self.b3_max
            range3 = min(cache_size, max_margin)
            outbucket3 =  [self.b3_cache.pop() for i in range(range3)]
        else:
            outbucket3 = []
        return outbucket0, outbucket1, outbucket2, outbucket3

    def mask_tokens( self, inputs, special_tokens_mask= None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.size(), self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
            #special_tokens_mask = special_tokens_mask.bool()

        #print(special_tokens_mask.size())
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.size(), 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.size(), 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.char2id.keys()), labels.size(), dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    
    def shortest_path_length(self, graph, max_node_num):
        # node_num = graph.number_of_nodes()
        sp_length = np.zeros([max_node_num, max_node_num])
        for node1, value in nx.shortest_path_length(graph):
            for node2, length in value.items():
                sp_length[node1][node2] = length
        # sp_length = torch.nn.functional.normalize(sp_length, dim=None)
        return sp_length
    
    def merge_to_3d(self, matrices):
        # Determine the maximum dimensions
        max_rows = max(mat.shape[0] for mat in matrices)
        max_cols = max(mat.shape[1] for mat in matrices)

        # Pad each matrix to the maximum dimensions and stack them
        padded_matrices = []
        for mat in matrices:
            padded_matrix = np.pad(mat, ((0, max_rows - mat.shape[0]), (0, max_cols - mat.shape[1])))
            padded_matrices.append(padded_matrix)

        # Convert the list of matrices to a 3D numpy array
        merged_3d = np.stack(padded_matrices)

        return merged_3d

    def smiles_to_hypergraph(self, batch_smiles):
        matrix_list = []
        if self.add_eos:
            eoffset = 1
        else:
            eoffset = 0
           
        if self.add_bos:
            boffset = 1
        else:
            boffset = 0
        
        for smiles_tokens in batch_smiles:
            smiles = "".join(smiles_tokens)
            mol = Chem.MolFromSmiles(smiles)

            # Identify rings
            ring_info = mol.GetRingInfo()
            rings = ring_info.AtomRings()

            # Create hypergraph
            hyperedges = collections.defaultdict(list)

            # Add rings to hypergraph
            for idx, ring in enumerate(rings):
                for atom in ring:
                    hyperedges[f"ring_{idx}"].append(atom)

            # Add functional groups to hypergraph
            for smarts in FUNCTIONAL_GROUPS:
                pattern = Chem.MolFromSmarts(smarts)
                matches = mol.GetSubstructMatches(pattern)
                for idx, match in enumerate(matches):
                    for atom in match:
                        hyperedges[f"{smarts}_{idx}"].append(atom)

            # Convert hypergraph to matrix
            num_atoms = mol.GetNumAtoms()
            num_hyperedges = len(hyperedges)
            matrix = np.zeros((num_atoms+boffset+eoffset, num_hyperedges), dtype=int)

            for idx, (key, atoms) in enumerate(hyperedges.items()):
                for atom in atoms:
                    matrix[atom+boffset][idx] = 1
            matrix_list.append(matrix)
        return torch.tensor(self.merge_to_3d(matrix_list), dtype=torch.int32)

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
        smiles = "".join(smiles_tokens)
        mol = Chem.MolFromSmiles(smiles)
        sidx = 0
        for i, atom in enumerate(mol.GetAtoms()):
            atom_symbol = atom.GetSymbol()
            while sidx < len(smiles_tokens):
                # transfrom SMILES to atom, match atom with graph's node
                smiles_atom = smiles_tokens[sidx].strip('[]').rstrip('+-0123456789').replace('@', '')
                if smiles_atom.endswith('H') and atom_symbol != 'He' and len(smiles_atom)>1:
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
        assert len(smiles_graph_map) == mol.GetNumAtoms()+boffset+eoffset, f'Error assert:(mol.GetNumAtoms()+boffset+eoffset):{mol.GetNumAtoms()+boffset+eoffset}, len(smiles_graph_map):{len(smiles_graph_map)}, smiles_tokens:{smiles_tokens}, smiles_graph_map:{smiles_graph_map}, graph:{[atom.GetSymbol() for atom in mol.GetAtoms()]}'
        assert len(graph_smiles_map) == mol.GetNumAtoms()+boffset+eoffset, print(len(graph_smiles_map), mol.GetNumAtoms()+boffset+eoffset)

        return smiles_graph_map, graph_smiles_map
    
    def smiles_to_crossAdj(self, smiles_tokens, graph, max_node_num, max_smiles_num):
        graph_pos = self.shortest_path_length(graph, max_node_num)
        
        _, graph_smiles_map = self.get_one_smiles_graph_map(smiles_tokens)
        if self.add_bos:
            boffset = 1
        else:
            boffset = 0
        if self.add_eos:
            eoffset = 1
        else:
            eoffset = 0
        graph2smiles_pos = np.zeros([max_node_num, max_smiles_num])
        graph2smiles_pos[:, graph_smiles_map] = graph_pos[:, :len(graph_smiles_map)]
        return graph2smiles_pos
        
        
    def smiles_to_2dgraph_padded(self, batch_smiles):
        batch_graph = []
        pos_list = []
        adj_list = []
        max_node_num = 0
        max_smiles_num = 0
            
        for smiles_tokens in batch_smiles:
            smiles = "".join(smiles_tokens)
            mol = Chem.MolFromSmiles(smiles)
            max_node_num = max(max_node_num, mol.GetNumAtoms())
            max_smiles_num = max(max_smiles_num, len(smiles_tokens))
        
        if self.add_bos == True:
            max_node_num += 1
            max_smiles_num += 1
            boffset = 1
        else:
            boffset = 0
            
        if self.add_eos == True:
            max_node_num += 1
            max_smiles_num += 1

        edge_index_l, edge_attr_l = [], []
        for smiles_tokens in batch_smiles:
            
            # smiles_tokens = self.regex.findall(smiles.strip('\n'))
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

            batch_graph.append(torch.tensor(graph_tokens))
            adj_list.append(torch.tensor(nx.adj_matrix(graph).toarray()))
            edge_index_l.append(torch.tensor(edge_index))
            edge_attr_l.append(torch.tensor(edge_attr))
        return batch_graph, adj_list, edge_index_l, edge_attr_l #pos_list
        
    def smiles_to_2dgraph(self, batch_smiles):
        batch_graph = []
        pos_list = []
        adj_list = []
        max_node_num = 0
        max_smiles_num = 0
            
        for smiles_tokens in batch_smiles:
            smiles = "".join(smiles_tokens)
            mol = Chem.MolFromSmiles(smiles)
            max_node_num = max(max_node_num, mol.GetNumAtoms())
            max_smiles_num = max(max_smiles_num, len(smiles_tokens))
        
        if self.add_bos == True:
            max_node_num += 1
            max_smiles_num += 1
            boffset = 1
        else:
            boffset = 0
            
        if self.add_eos == True:
            max_node_num += 1
            max_smiles_num += 1

        edge_index_l, edge_attr_l = [], []
        for smiles_tokens in batch_smiles:
            smiles = "".join(smiles_tokens)
            node_list = []
            mol = Chem.MolFromSmiles(smiles)
            graph = nx.Graph()
                
            if self.add_bos == True:
                node_list = [self.graph_char2id['<bos>']] + node_list
                graph.add_node(0)
                
            for atom in mol.GetAtoms():
                node_list.append(self.graph_char2id[atom.GetSymbol()])
                graph.add_node(atom.GetIdx()+boffset, type=atom.GetAtomicNum())
                
            edge_index, edge_attr = [[], []], []
            for bond in mol.GetBonds():
                start_idx = bond.GetBeginAtomIdx() + boffset
                end_idx = bond.GetEndAtomIdx() + boffset
                graph.add_edge(start_idx, end_idx, bond_type=bond.GetBondTypeAsDouble())
                
                edge_index[0].append(start_idx)
                edge_index[1].append(end_idx)
                bond_dir = bond.GetBondDir()
                bond_dir_id = bond_vocab_dict['bond_dir'][bond_dir] + 2 # 0: self-loops; 1: masking
                bond_type = bond.GetBondType()
                bond_type_id = bond_vocab_dict['bond_type'][bond_type] + 2 # 0: self-loops; 1: masking
                edge_attr.append([bond_dir_id, bond_type_id])

            if self.add_eos == True:
                node_list = node_list + [self.graph_char2id['<eos>']]
                graph.add_node(graph.number_of_nodes())

            batch_graph.append(torch.tensor(node_list))
            adj_list.append(torch.tensor(nx.adjacency_matrix(graph).toarray()))
            edge_index_l.append(torch.tensor(edge_index))
            edge_attr_l.append(torch.tensor(edge_attr))

        pos_list = torch.from_numpy(np.array(pos_list)).int()
        return batch_graph, adj_list, edge_index_l, edge_attr_l #pos_list

    def get_smiles_graph_map(self, tokens):
        if self.add_bos:
            boffset = 1
        else:
            boffset = 0
            
        smiles_graph_map_list = []
        graph_smiles_map_list = []
        for smiles_tokens in tokens: 
            # ['C', 'C', 'O', 'c', '1', 'c', 'c', '(', 'C', '=', 'N', 'N', 'C', '(', '=', 'O', ')', ...]
            smiles_graph_map = collections.defaultdict(list)
            graph_smiles_map = collections.defaultdict(list)
            if self.add_bos:
                smiles_graph_map[0] = 0 # bos=0
                graph_smiles_map[0] = 0
            smiles = "".join(smiles_tokens)
            mol = Chem.MolFromSmiles(smiles)
            sidx = 0
            for i, atom in enumerate(mol.GetAtoms()):
                atom_symbol = atom.GetSymbol()
                while sidx < len(smiles_tokens):
                    # transfrom SMILES to atom, match atom with graph's node
                    smiles_atom = smiles_tokens[sidx].strip('[]').rstrip('+-0123456789').replace('@', '')
                    if smiles_atom.endswith('H') and atom_symbol != 'He' and len(smiles_atom)>1:
                        smiles_atom = smiles_atom[:-1]
                    if smiles_atom.upper() == atom_symbol.upper():
                        smiles_graph_map[sidx+boffset] = i+boffset
                        graph_smiles_map[i+boffset] = sidx+boffset
                        sidx += 1
                        break
                    else:
                        sidx += 1
            if self.add_eos:
                smiles_graph_map[len(smiles_tokens)+boffset] = mol.GetNumAtoms()+boffset
            assert mol.GetNumAtoms()==(len(smiles_graph_map)-int(self.add_eos)-int(self.add_bos)), 'Error assert:(i+1):{a}, len(smiles_graph_map):{length}, smiles_tokens:{smiles_tokens}, smiles_graph_map:{smiles_graph_map}, graph:{graph}'.format(a=i+1, length=len(smiles_graph_map), smiles_tokens=smiles_tokens, smiles_graph_map=smiles_graph_map, graph=[atom.GetSymbol() for atom in mol.GetAtoms()])

            smiles_graph_map_list.append(smiles_graph_map)
            graph_smiles_map_list.append(graph_smiles_map)
        return smiles_graph_map_list, graph_smiles_map_list
    
    def transform_masked_labels(self, map_list, input_array, input_labels, masked_indices, indices_replaced, indices_random=None):
        true_positions = (~masked_indices).nonzero().tolist()
        for i, j in true_positions:
            input_labels[i, map_list[i][j]] = -100
        input_labels[input_labels==self.graph_pad] = -100
            
        true_positions = (indices_replaced).nonzero().tolist()
        for i, j in true_positions:
            input_array[i, map_list[i][j]] = self.graph_mask
            
        if indices_random != None: # generate graph masked labels based on smiles input
            random_words = torch.randint(len(self.graph_char2id.keys()), input_labels.size(), dtype=torch.long)
            true_positions = (indices_random).nonzero().tolist()
            for i, j in true_positions:
                input_array[i, map_list[i][j]] = random_words[i, map_list[i][j]]
            
        return input_array, input_labels

    def get_smiles_substructure(self, smiles_array):
        result = []
        max_len = 0
        for smiles_tokens in smiles_array:
            special_substructure = []
            matched_atoms = []

            smiles = "".join(smiles_tokens)
            mol = Chem.MolFromSmiles(smiles)

            # Identify rings
            ring_info = mol.GetRingInfo()
            rings = ring_info.AtomRings()

            recognized_atoms = set()

            # Collect atoms from rings
            for ring in rings:
                recognized_atoms.update(ring)

            # Add rings to hypergraph
            special_substructure += [list(ring) for ring in rings]

            # Add functional groups to hypergraph
            for smarts in FUNCTIONAL_GROUPS:
                pattern = Chem.MolFromSmarts(smarts)    
                matches = mol.GetSubstructMatches(pattern)
                special_substructure += [list(m) for m in matches]
                # Collect atoms from functional groups
                for match in matches:
                    recognized_atoms.update(match)
                    
            
            all_atom_indices = set(range(mol.GetNumAtoms()))
            unrecognized_atoms = all_atom_indices - recognized_atoms
            special_substructure += [[atom] for atom in unrecognized_atoms]
            result.append(special_substructure)
            max_len = max(max_len, len(special_substructure))
        mask = torch.zeros(len(smiles_array), max_len, dtype=torch.bool)
        for i, sub in enumerate(result):
            mask[i][len(sub):] = 1
        return result, mask
    
    def get_smiles_subsequence(self, smiles_array):
        result = []
        max_len = 0
        for smiles_tokens in smiles_array:
            special_substructure = []
            matched_atoms = []

            smiles = "".join(smiles_tokens)
            mol = Chem.MolFromSmiles(smiles)

            # Identify rings
            ring_info = mol.GetRingInfo()
            rings = ring_info.AtomRings()

            # Add rings to hypergraph
            special_substructure += [list(ring) for ring in rings]

            # Add functional groups to hypergraph
            for smarts in FUNCTIONAL_GROUPS:
                pattern = Chem.MolFromSmarts(smarts)    
                matches = mol.GetSubstructMatches(pattern)
                special_substructure += [list(m) for m in matches]
            
            recognized_atoms = set()
            subsequence_l = []
            max_num = mol.GetNumAtoms()
            for substructure in special_substructure:
                length = len(substructure)
                start_idx = random.randint(0, max_num-length)
                subsequence = list(range(start_idx, start_idx+length))
                subsequence_l.append(subsequence)
                recognized_atoms.update(set(subsequence))
            
            all_atom_indices = set(range(mol.GetNumAtoms()))
            unrecognized_atoms = all_atom_indices - recognized_atoms
            subsequence_l += [[atom] for atom in unrecognized_atoms]
            result.append(subsequence_l)
            max_len = max(max_len, len(subsequence_l))
        mask = torch.zeros(len(smiles_array), max_len, dtype=torch.bool)
        for i, sub in enumerate(result):
            mask[i][len(sub):] = 1
        return result, mask

    def mask_smiles_graph_tokens_substructure(self, tokens, smiles_array, graph_array, smiles_special_tokens_mask, graph_special_tokens_mask):
        smiles_labels = smiles_array.clone()
        graph_labels = graph_array.clone()
        
        # map graph to smiles
        _, graph_smiles_map_list = self.get_smiles_graph_map(tokens)
        
        # get graph substructure
        substructure, substructure_mask = self.get_smiles_substructure(tokens)
        
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(substructure_mask.size(), self.mlm_probability)

        # Graph label mask
        probability_matrix.masked_fill_(substructure_mask, value=0.0)
        masked_substruture = torch.bernoulli(probability_matrix).bool()
        masked_substruture_idx = torch.nonzero(masked_substruture, as_tuple=True)
        
        masked_node_idx = ([],[])
        for i, j in zip(masked_substruture_idx[0].tolist(), masked_substruture_idx[1].tolist()):
            node_idx_list = substructure[i][j]
            for node_idx in node_idx_list:
                masked_node_idx[0].append(i)
                masked_node_idx[1].append(node_idx)
        if len(masked_node_idx[0]) > 0:
            masked_indices = torch.zeros_like(graph_array, dtype=torch.bool)
            masked_indices = masked_indices.index_put((torch.tensor(masked_node_idx[0], dtype=torch.long), torch.tensor(masked_node_idx[1], dtype=torch.long)), torch.tensor(1, dtype=torch.bool))    
        else:
            print('no subgrpah masked, choose to mask nodes')
            
            if graph_special_tokens_mask is None:
                graph_special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in graph_labels.tolist()
                ]
                graph_special_tokens_mask = torch.tensor(graph_special_tokens_mask, dtype=torch.bool)
            else:
                graph_special_tokens_mask = torch.tensor(graph_special_tokens_mask, dtype=torch.bool)
            probability_matrix = torch.full(graph_labels.size(), self.mlm_probability)
            probability_matrix.masked_fill_(graph_special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
        
        graph_labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(graph_labels.size(), 1.0)).bool() & masked_indices
        graph_array[indices_replaced] = self.mask
        
        # smiles label mask via smiles_graph_map_list
        smiles_arrays, smiles_labels = self.transform_masked_labels(graph_smiles_map_list, smiles_array, smiles_labels, masked_indices, indices_replaced)
        
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return smiles_arrays, smiles_labels, graph_array, graph_labels
    
    def mask_smiles_graph_tokens(self, tokens, smiles_array, graph_array, smiles_special_tokens_mask, graph_special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        # map smiles to graph
        smiles_graph_map_list, _ = self.get_smiles_graph_map(tokens)
        
        smiles_labels = smiles_array.clone()
        graph_labels = graph_array.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(smiles_labels.size(), self.mlm_probability)
        if smiles_special_tokens_mask is None:
            smiles_special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in smiles_labels.tolist()
            ]
            smiles_special_tokens_mask = torch.tensor(smiles_special_tokens_mask, dtype=torch.bool)
        else:
            smiles_special_tokens_mask = torch.tensor(smiles_special_tokens_mask, dtype=torch.bool)
            
        if graph_special_tokens_mask is None:
            graph_special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in graph_labels.tolist()
            ]
            graph_special_tokens_mask = torch.tensor(graph_special_tokens_mask, dtype=torch.bool)
        else:
            graph_special_tokens_mask = torch.tensor(graph_special_tokens_mask, dtype=torch.bool)
            #special_tokens_mask = special_tokens_mask.bool()

        # SMILES label mask
        probability_matrix.masked_fill_(smiles_special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        smiles_labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(smiles_labels.size(), 0.8)).bool() & masked_indices
        smiles_array[indices_replaced] = self.mask

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(smiles_labels.size(), 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.char2id.keys()), smiles_labels.size(), dtype=torch.long)
        smiles_array[indices_random] = random_words[indices_random]

        # graph label mask via smiles_graph_map_list
        graph_arrays, graph_labels = self.transform_masked_labels(smiles_graph_map_list, graph_array, graph_labels, masked_indices, indices_replaced, indices_random)
        
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return smiles_array, smiles_labels, graph_arrays, graph_labels


    def mask_smiles_graph_tokens_subsequence(self, tokens, smiles_array, graph_array, smiles_special_tokens_mask, graph_special_tokens_mask):
        smiles_labels = smiles_array.clone()
        graph_labels = graph_array.clone()
        
        # map graph to smiles
        _, graph_smiles_map_list = self.get_smiles_graph_map(tokens)
        
        # get graph subsequence
        subsequence, subsequence_mask = self.get_smiles_subsequence(tokens)
        
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(subsequence_mask.size(), self.mlm_probability)

        # Graph label mask
        probability_matrix.masked_fill_(subsequence_mask, value=0.0)
        masked_subsequence = torch.bernoulli(probability_matrix).bool()
        masked_subsequence_idx = torch.nonzero(masked_subsequence, as_tuple=True)
        
        masked_node_idx = ([],[])
        for i, j in zip(masked_subsequence_idx[0].tolist(), masked_subsequence_idx[1].tolist()):
            node_idx_list = subsequence[i][j]
            for node_idx in node_idx_list:
                masked_node_idx[0].append(i)
                masked_node_idx[1].append(node_idx)
        if len(masked_node_idx[0]) > 0:
            masked_indices = torch.zeros_like(graph_array, dtype=torch.bool)
            masked_indices = masked_indices.index_put((torch.tensor(masked_node_idx[0], dtype=torch.long), torch.tensor(masked_node_idx[1], dtype=torch.long)), torch.tensor(1, dtype=torch.bool))    
        else:
            print('no subsequence masked, choose to mask nodes')
            
            if graph_special_tokens_mask is None:
                graph_special_tokens_mask = [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in graph_labels.tolist()
                ]
                graph_special_tokens_mask = torch.tensor(graph_special_tokens_mask, dtype=torch.bool)
            else:
                graph_special_tokens_mask = torch.tensor(graph_special_tokens_mask, dtype=torch.bool)
            probability_matrix = torch.full(graph_labels.size(), self.mlm_probability)
            probability_matrix.masked_fill_(graph_special_tokens_mask, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            
        
        graph_labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        print('masked functional groups ratio:', torch.sum(masked_indices)/masked_indices.nelement())
        print('masked atoms ratio:', torch.sum(masked_subsequence)/masked_subsequence.nelement())
        
        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(graph_labels.size(), 1.0)).bool() & masked_indices
        graph_array[indices_replaced] = self.mask
        
        # smiles label mask via smiles_graph_map_list
        smiles_arrays, smiles_labels = self.transform_masked_labels(graph_smiles_map_list, smiles_array, smiles_labels, masked_indices, indices_replaced)
        
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return smiles_arrays, smiles_labels, graph_array, graph_labels
    
    
    def pack_smiles_graph_tensors(self, tokens):
        smiles_array = self.encoder(tokens) # tokens to ids
        smiles_array =  torch.nn.utils.rnn.pad_sequence(smiles_array, batch_first=True, padding_value=self.pad)
        smiles_special_tokens_mask = [list(map(lambda x: 1 if x in [self.bos, self.eos, self.pad] else 0, stuff)) for stuff in smiles_array.tolist()]
        
        if self.graph_padding == 'False':
            graph_array, twod_position_embedding, edge_index_l, edge_attr_l = self.smiles_to_2dgraph(tokens)
        elif self.graph_padding == 'True':
            graph_array, twod_position_embedding, edge_index_l, edge_attr_l = self.smiles_to_2dgraph_padded(tokens)
            
        # pad adj
        max_twod_pos_emb_dim1 = max([t.size(0) for t in twod_position_embedding])
        max_twod_pos_emb_dim2 = max([t.size(1) for t in twod_position_embedding])
        twod_position_embeddings_padded = torch.full((len(twod_position_embedding), max_twod_pos_emb_dim1, max_twod_pos_emb_dim2), 0)
        for i, t in enumerate(twod_position_embedding):
            twod_position_embeddings_padded[i, :t.size(0), :t.size(1)] = t
            
        # process edge_index, edge_attr
        g_node_num = len(graph_array[0])
        edge_index_merged = []
        for i in range(len(graph_array)):
            edge_index_merged.append(edge_index_l[i]+g_node_num*i)
        edge_index_batch = torch.cat(edge_index_merged, dim=1).long()
        edge_attr_batch = torch.cat(edge_attr_l, dim=0).int()
            
        # hyper-graph
        hypergraph_adj = self.smiles_to_hypergraph(tokens)
        graph_array = torch.nn.utils.rnn.pad_sequence(graph_array, batch_first=True, padding_value=self.pad)
        graph_special_tokens_mask = [list(map(lambda x: 1 if x in [self.graph_bos, self.graph_eos, self.graph_pad] else 0, stuff)) for stuff in graph_array.tolist()]
        masked_smiles_array, masked_smiles_labels, masked_graph_array, masked_graph_labels = self.mask_smiles_graph_tokens(tokens, smiles_array, graph_array, smiles_special_tokens_mask, graph_special_tokens_mask)
        
        return masked_smiles_array, masked_smiles_labels, masked_graph_array, masked_graph_labels, \
                twod_position_embeddings_padded, hypergraph_adj, edge_index_batch, edge_attr_batch
        
    
    def pack_smiles_graph_tensors_substructure(self, tokens):
        smiles_array = self.encoder(tokens) # tokens to ids
        smiles_array =  torch.nn.utils.rnn.pad_sequence(smiles_array, batch_first=True, padding_value=self.pad)
        smiles_special_tokens_mask = [list(map(lambda x: 1 if x in [self.bos, self.eos, self.pad] else 0, stuff)) for stuff in smiles_array.tolist()]
        
        if self.graph_padding == 'False':
            graph_array, twod_position_embedding, edge_index_l, edge_attr_l = self.smiles_to_2dgraph(tokens)
        elif self.graph_padding == 'True':
            graph_array, twod_position_embedding, edge_index_l, edge_attr_l = self.smiles_to_2dgraph_padded(tokens)
            
        # pad adj
        max_twod_pos_emb_dim1 = max([t.size(0) for t in twod_position_embedding])
        max_twod_pos_emb_dim2 = max([t.size(1) for t in twod_position_embedding])
        twod_position_embeddings_padded = torch.full((len(twod_position_embedding), max_twod_pos_emb_dim1, max_twod_pos_emb_dim2), 0)
        for i, t in enumerate(twod_position_embedding):
            twod_position_embeddings_padded[i, :t.size(0), :t.size(1)] = t
            
        # process edge_index, edge_attr
        g_node_num = len(graph_array[0])
        edge_index_merged = []
        for i in range(len(graph_array)):
            edge_index_merged.append(edge_index_l[i]+g_node_num*i)
        edge_index_batch = torch.cat(edge_index_merged, dim=1).long()
        edge_attr_batch = torch.cat(edge_attr_l, dim=0).int()
        
        # hyper-graph
        hypergraph_adj = self.smiles_to_hypergraph(tokens)
        graph_array = torch.nn.utils.rnn.pad_sequence(graph_array, batch_first=True, padding_value=self.pad)
        graph_special_tokens_mask = [list(map(lambda x: 1 if x in [self.graph_bos, self.graph_eos, self.graph_pad] else 0, stuff)) for stuff in graph_array.tolist()]
        masked_smiles_array, masked_smiles_labels, masked_graph_array, masked_graph_labels = self.mask_smiles_graph_tokens_substructure(tokens, smiles_array, graph_array, smiles_special_tokens_mask, graph_special_tokens_mask)
        return masked_smiles_array, masked_smiles_labels, masked_graph_array, masked_graph_labels,  \
                twod_position_embeddings_padded, hypergraph_adj, edge_index_batch, edge_attr_batch
    
    def pack_smiles_graph_tensors_subsequence(self, tokens):
        smiles_array = self.encoder(tokens) # tokens to ids
        smiles_array =  torch.nn.utils.rnn.pad_sequence(smiles_array, batch_first=True, padding_value=self.pad)
        smiles_special_tokens_mask = [list(map(lambda x: 1 if x in [self.bos, self.eos, self.pad] else 0, stuff)) for stuff in smiles_array.tolist()]
        
        if self.graph_padding == 'False':
            graph_array, twod_position_embedding, edge_index_l, edge_attr_l = self.smiles_to_2dgraph(tokens)
        elif self.graph_padding == 'True':
            graph_array, twod_position_embedding, edge_index_l, edge_attr_l = self.smiles_to_2dgraph_padded(tokens)
            
        # pad adj
        max_twod_pos_emb_dim1 = max([t.size(0) for t in twod_position_embedding])
        max_twod_pos_emb_dim2 = max([t.size(1) for t in twod_position_embedding])
        twod_position_embeddings_padded = torch.full((len(twod_position_embedding), max_twod_pos_emb_dim1, max_twod_pos_emb_dim2), 0)
        for i, t in enumerate(twod_position_embedding):
            twod_position_embeddings_padded[i, :t.size(0), :t.size(1)] = t
            
        # process edge_index, edge_attr
        g_node_num = len(graph_array[0])
        edge_index_merged = []
        for i in range(len(graph_array)):
            edge_index_merged.append(edge_index_l[i]+g_node_num*i)
        edge_index_batch = torch.cat(edge_index_merged, dim=1).long()
        edge_attr_batch = torch.cat(edge_attr_l, dim=0).int()
        
        # hyper-graph
        hypergraph_adj = self.smiles_to_hypergraph(tokens)
        graph_array = torch.nn.utils.rnn.pad_sequence(graph_array, batch_first=True, padding_value=self.pad)
        graph_special_tokens_mask = [list(map(lambda x: 1 if x in [self.graph_bos, self.graph_eos, self.graph_pad] else 0, stuff)) for stuff in graph_array.tolist()]
        masked_smiles_array, masked_smiles_labels, masked_graph_array, masked_graph_labels = self.mask_smiles_graph_tokens_subsequence(tokens, smiles_array, graph_array, smiles_special_tokens_mask, graph_special_tokens_mask)
        return masked_smiles_array, masked_smiles_labels, masked_graph_array, masked_graph_labels,  \
                twod_position_embeddings_padded, hypergraph_adj, edge_index_batch, edge_attr_batch
    
    def pack_smiles_graph_tensors_merged(self, tokens):
        smiles_array = self.encoder(tokens) # tokens to ids
        smiles_array =  torch.nn.utils.rnn.pad_sequence(smiles_array, batch_first=True, padding_value=self.pad)
        smiles_special_tokens_mask = [list(map(lambda x: 1 if x in [self.bos, self.eos, self.pad] else 0, stuff)) for stuff in smiles_array.tolist()]
        
        if self.graph_padding == 'False':
            graph_array, twod_position_embedding, edge_index_l, edge_attr_l = self.smiles_to_2dgraph(tokens)
        elif self.graph_padding == 'True':
            graph_array, twod_position_embedding, edge_index_l, edge_attr_l = self.smiles_to_2dgraph_padded(tokens)
        # pad adj
        max_twod_pos_emb_dim1 = max([t.size(0) for t in twod_position_embedding])
        max_twod_pos_emb_dim2 = max([t.size(1) for t in twod_position_embedding])
        twod_position_embeddings_padded = torch.full((len(twod_position_embedding), max_twod_pos_emb_dim1, max_twod_pos_emb_dim2), 0)
        for i, t in enumerate(twod_position_embedding):
            twod_position_embeddings_padded[i, :t.size(0), :t.size(1)] = t
            
        # process edge_index, edge_attr
        g_node_num = len(graph_array[0])
        edge_index_merged = []
        for i in range(len(graph_array)):
            edge_index_merged.append(edge_index_l[i]+g_node_num*i)
        edge_index_batch = torch.cat(edge_index_merged, dim=1).long()
        edge_attr_batch = torch.cat(edge_attr_l, dim=0).int()
        
        # hyper-graph
        hypergraph_adj = self.smiles_to_hypergraph(tokens)
        graph_array = torch.nn.utils.rnn.pad_sequence(graph_array, batch_first=True, padding_value=self.pad)
        graph_special_tokens_mask = [list(map(lambda x: 1 if x in [self.graph_bos, self.graph_eos, self.graph_pad] else 0, stuff)) for stuff in graph_array.tolist()]
        masked_smiles_array1, masked_smiles_labels1, masked_graph_array1, masked_graph_labels1 = self.mask_smiles_graph_tokens_substructure(tokens, smiles_array, graph_array, smiles_special_tokens_mask, graph_special_tokens_mask)
        masked_smiles_array2, masked_smiles_labels2, masked_graph_array2, masked_graph_labels2 = self.mask_smiles_graph_tokens(tokens, smiles_array, graph_array, smiles_special_tokens_mask, graph_special_tokens_mask)
        
        return [(masked_smiles_array1, masked_smiles_labels1, masked_graph_array1, masked_graph_labels1,  \
                 twod_position_embeddings_padded, hypergraph_adj, edge_index_batch, edge_attr_batch), 
                (masked_smiles_array2, masked_smiles_labels2, masked_graph_array2, masked_graph_labels2,  \
                 twod_position_embeddings_padded, hypergraph_adj, edge_index_batch, edge_attr_batch)]
    
    def pack_tensors(self, tokens):
        if self.pretrain_style == 'atom':
            return [self.pack_smiles_graph_tensors(tokens)]
        elif self.pretrain_style == 'subgraph':
            return [self.pack_smiles_graph_tensors_substructure(tokens)]
        elif self.pretrain_style == 'subgraph_and_atom':
            return self.pack_smiles_graph_tensors_merged(tokens)
        elif self.pretrain_style == 'subsequence':
            return [self.pack_smiles_graph_tensors_subsequence(tokens)]
        else:
            raise ValueError('Error config.pretrain_style:{}'.format(self.pretrain_style))
        
    def process(self, text):
        print('b0_cache:{}, b1_cache:{}, b2_cache:{}, b3_cache:{}'.format(len(self.b0_cache), len(self.b1_cache), len(self.b2_cache), len(self.b3_cache)))
        #lengths = []
        arrays, targets, position_embedding, hypergraphs, edge_index_l, edge_attr_l = [], [], [], [], [], []
        idx = 0
        for tokens in self.process_text(text): # divede SMILES into 4types based on SMILES's length
            idx += 1
            if len(tokens) > 0:
                for record in self.pack_tensors(tokens):
                    smiles_array, smiles_target, graph_array, graph_target, twoD_position_embedding, hypergraph_adj, edge_index_batch, edge_attr_batch = record
                    arrays.append((smiles_array, graph_array))
                    targets.append((smiles_target, graph_target))
                    position_embedding.append(twoD_position_embedding)
                    hypergraphs.append(hypergraph_adj)
                    edge_index_l.append(edge_index_batch)
                    edge_attr_l.append(edge_attr_batch)
        flag = True
        while flag == True:
            flag = False
            for tokens in self.process_cache():
                idx += 1
                if len(tokens) > 0:
                    flag = True
                    for record in self.pack_tensors(tokens):
                        smiles_array, smiles_target, graph_array, graph_target, twoD_position_embedding, hypergraph_adj, edge_index_batch, edge_attr_batch = record
                        arrays.append((smiles_array, graph_array))
                        targets.append((smiles_target, graph_target))
                        position_embedding.append(twoD_position_embedding)
                        hypergraphs.append(hypergraph_adj)
                        edge_index_l.append(edge_index_batch)
                        edge_attr_l.append(edge_attr_batch)
            
        return arrays, targets, position_embedding, hypergraphs, edge_index_l, edge_attr_l
