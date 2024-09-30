from math import sqrt
import numpy as np
import networkx as nx
import torch

from rdkit import Chem
from torch.nn import Linear, Module, Dropout, Embedding
from fast_transformers.attention import AttentionLayer
from fast_transformers.events import EventDispatcher, QKVEvent

from fast_transformers.attention_registry import AttentionRegistry, Optional, Callable, Int, \
    EventDispatcherInstance, Float, RecurrentAttentionRegistry
from fast_transformers.events import EventDispatcher, AttentionEvent
from fast_transformers.feature_maps import elu_feature_map
from fast_transformers.recurrent._utils import check_state
from model.position_encoding import GraphPositionalEncoding, SmilesPositionalEncoding
from .position_encoding import RotaryEmbedding, apply_rotary_pos_emb


class RotateAttentionLayer(AttentionLayer):
    """Rotate attention layer inherits from fast_transformer attention layer. 
        The only thing added is an Embedding encoding, for more information
        on the attention layer see the fast_transformers code
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, event_dispatcher=""):
        super(RotateAttentionLayer, self).__init__(attention,d_model, n_heads, d_keys=d_keys,
                 d_values=d_values, event_dispatcher=event_dispatcher)

        self.rotaryemb = RotaryEmbedding(d_keys)
        print('Using Rotation Embedding')

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """
        Using the same frame work as the fast_Transformers attention layer
        but injecting rotary information to the queries and the keys
        after the keys and queries are projected. 
        In the argument description we make use of the following sizes

            - N: the batch size
            - L: The maximum length of the queries
            - S: The maximum length of the keys (the actual length per sequence
              is given by the length mask)
            - D: The input feature dimensionality passed in the constructor as
              'd_model'

        Arguments
        ---------
            queries: (N, L, D) The tensor containing the queries
            keys: (N, S, D) The tensor containing the keys
            values: (N, S, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of

        Returns
        -------
            The new value for each query as a tensor of shape (N, L, D).
        """
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        cos, sin = self.rotaryemb(queries)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        values = self.value_projection(values).view(N, S, H, -1)
        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))

        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)


class CrossAttentionLinearRotaryLayer(AttentionLayer):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, event_dispatcher=""):
        super(CrossAttentionLinearRotaryLayer, self).__init__(attention, d_model, n_heads, d_keys=d_keys,
                 d_values=d_values, event_dispatcher=event_dispatcher)
        
        self.query_projection = None
        self.key_projection = None
        self.value_projection = None
        self.out_projection = None
        self.out_projection1 = Linear(d_values * n_heads, d_model)
        self.out_projection2 = Linear(d_values * n_heads, d_model)
        self.query_projection1 = Linear(d_model, d_keys * n_heads)
        self.query_projection2 = Linear(d_model, d_keys * n_heads)
        self.key_projection1 = Linear(d_model, d_keys * n_heads)
        self.key_projection2 = Linear(d_model, d_keys * n_heads)
        self.value_projection1 = Linear(d_model, d_values * n_heads)
        self.value_projection2 = Linear(d_model, d_values * n_heads)

        self.rotaryemb = RotaryEmbedding(d_keys)
        print('Using CrossAttentionLinearRotaryLayer')
        
    def forward(self, queries1, keys1, values1, queries2, keys2, values2, 
                attn_mask1, query_lengths1, key_lengths1, pos1, 
                attn_mask2, query_lengths2, key_lengths2, pos2):
        # Extract the dimensions into local variables
        N1, L1, _ = queries1.shape
        _, S1, _ = keys1.shape
        N2, L2, _ = queries2.shape
        _, S2, _ = keys2.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries1 = self.query_projection1(queries1).view(N1, L1, H, -1)
        keys1 = self.key_projection1(keys1).view(N1, S1, H, -1)
        values1 = self.value_projection1(values1).view(N1, S1, H, -1)

        # Project the queries/keys/values
        queries2 = self.query_projection2(queries2).view(N2, L2, H, -1)
        keys2 = self.key_projection2(keys2).view(N2, S2, H, -1)
        values2 = self.value_projection2(values2).view(N2, S2, H, -1)
        
        cos, sin = self.rotaryemb(queries1)
        queries1, keys1 = apply_rotary_pos_emb(queries1, keys1, cos, sin)
        cos, sin = self.rotaryemb(queries2)
        queries2, keys2 = apply_rotary_pos_emb(queries2, keys2, cos, sin)
        # print('query2 no rotary embedding')
        
        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries1, keys1, values1))
        self.event_dispatcher.dispatch(QKVEvent(self, queries2, keys2, values2))
        
        # pos = torch.tensor(pos)
        # Compute the attention
        # # print('self.inner_attention', self.inner_attention)
        new_values1, new_values2 = self.inner_attention(
            queries1,
            keys1,
            values1,
            queries2,
            keys2,
            values2,
            attn_mask1,
            query_lengths1,
            key_lengths1,
            pos1,
            attn_mask2,
            query_lengths2,
            key_lengths2,
            pos2
        )
        new_values1 = new_values1.view(N1, L1, -1)
        new_values2 = new_values2.view(N2, L2, -1)
        print('new_values1', new_values1.shape)
        print('new_values2', new_values2.shape)
        
        # Project the output and return
        return self.out_projection1(new_values1), self.out_projection2(new_values2)
    

class LinearAttention_Cross(Module):
    """Implement the scaled dot product attention with softmax.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, query_dimensions, feature_map=None, eps=1e-6,
                 event_dispatcher=""):
        super(LinearAttention_Cross, self).__init__()
        print('Using LinearAttention_Cross')
        # self.softmax_temp = softmax_temp
        # self.dropout = Dropout(attention_dropout)
        # self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        self.feature_map = (
            feature_map(query_dimensions) if feature_map else
            elu_feature_map(query_dimensions)
        )
        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries1, keys1, values1, queries2, keys2, values2, 
                attn_mask1, query_lengths1, key_lengths1, pos1,
                attn_mask2, query_lengths2, key_lengths2, pos2):
        """Implements the multihead softmax attention.

        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # Extract some shapes and compute the temperature
        self.feature_map.new_feature_map(queries1.device)
        Q1 = self.feature_map.forward_queries(queries1)
        K1 = self.feature_map.forward_keys(keys1)
        Q2 = self.feature_map.forward_queries(queries2)
        K2 = self.feature_map.forward_keys(keys2)
        
        if not attn_mask1.all_ones or not attn_mask2.all_ones:
            raise RuntimeError(("LinearAttention does not support arbitrary "
                                "attention masks"))
        K1 = K1 * key_lengths1.float_matrix[:, :, None, None]
        K2 = K2 * key_lengths2.float_matrix[:, :, None, None]
        
        KV1 = torch.einsum("nshd,nshm->nhmd", K2, values2) 
        Z1 = 1/(torch.einsum("nlhd,nhd->nlh", Q1, K2.sum(dim=1))+self.eps) 
        V1 = torch.einsum("nlhd,nhmd,nlh->nlhm", Q1, KV1, Z1)
        
        KV2 = torch.einsum("nshd,nshm->nhmd", K1, values1)
        Z2 = 1/(torch.einsum("nlhd,nhd->nlh", Q2, K1.sum(dim=1))+self.eps)
        V2 = torch.einsum("nlhd,nhmd,nlh->nlhm", Q2, KV2, Z2)
        
#         KV1 = torch.einsum("nshd,nshm->nhmd", K1, values1)
#         Z1 = 1/(torch.einsum("nlhd,nhd->nlh", Q1, K1.sum(dim=1))+self.eps)
#         V1 = torch.einsum("nlhd,nhmd,nlh->nlhm", Q1, KV1, Z1)
        
#         KV2 = torch.einsum("nshd,nshm->nhmd", K2, values2) 
#         Z2 = 1/(torch.einsum("nlhd,nhd->nlh", Q2, K2.sum(dim=1))+self.eps) 
#         V2 = torch.einsum("nlhd,nhmd,nlh->nlhm", Q2, KV2, Z2)
        
        # Make sure that what we return is contiguous
        return V1.contiguous(), V2.contiguous()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "linear_attention_cross", LinearAttention_Cross,
    [
        ("query_dimensions", Int),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
    
    
# --------------------------------------------------------------------------------------------------------------------------------------------------

class FullAttention_Smiles(Module):
    """Implement the scaled dot product attention with softmax.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, softmax_temp=None, attention_dropout=0.1,
                 event_dispatcher=""):
        super(FullAttention_Smiles, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths, pos=None):
        """Implements the multihead softmax attention.

        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # # print('FullAttention_Smiles')
        # Extract some shapes and compute the temperature
        # print('FullAttention_Smiles queries.shape', queries.shape)
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        
        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhe,nshe->nhls", queries, keys)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        QK = QK + key_lengths.additive_matrix[:, None, None]

        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(softmax_temp * QK, dim=-1))
        V = torch.einsum("nhls,nshd->nlhd", A, values)
        
        # tmp = 0
        # for tmp_m in pos:
        #     tmp = max(tmp, len(tmp_m))
        
        # # print('full_attention_spatial | queries:{} | keys:{} | values:{} | QK:{} | A:{} | pos:{} | max(pos[i]):{} | padding_pos:{}'.format(queries.size(), keys.size(), values.size(), QK.size(), A.size(), len(pos), 0, padding_pos.size()))
        # full_attention_spatial | queries:torch.Size([150, 92, 12, 64]) | keys:torch.Size([150, 92, 12, 64]) | values:torch.Size([150, 92, 12, 64]) | QK:torch.Size([150, 12, 92, 92]) | A:torch.Size([150, 12, 92, 92]) | pos:150 | max(pos[i]):92
        # full_attention_spatial | queries:torch.Size([50, 201, 12, 64]) | keys:torch.Size([50, 201, 12, 64]) | values:torch.Size([50, 201, 12, 64]) | QK:torch.Size([50, 12, 201, 201]) | A:torch.Size([50, 12, 201, 201])
        # Let the world know of the attention matrix
        self.event_dispatcher.dispatch(AttentionEvent(self, A))

        # Make sure that what we return is contiguous
        return V.contiguous()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "full_attention_smiles", FullAttention_Smiles,
    [
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, "")),
    ]
)

class FullAttention_Spatial(Module):
    """Implement the scaled dot product attention with softmax.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, softmax_temp=None, attention_dropout=0.1,
                 event_dispatcher="", num_heads=None):
        super(FullAttention_Spatial, self).__init__()
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        pos_encoding_layer = GraphPositionalEncoding
        # # print('pos_encoding_layer', pos_encoding_layer)
        # # print('n_heads', num_heads)
        self.pos_encoding_layer = pos_encoding_layer(n_heads=num_heads, pos_type='adj_matrix')

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths, pos=None):
        """Implements the multihead softmax attention.

        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # print('In FullAttention_Spatial')
        # Extract some shapes and compute the temperature
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        
        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhe,nshe->nhls", queries, keys)
        if not attn_mask.all_ones:
            QK = QK + attn_mask.additive_matrix
        QK = QK + key_lengths.additive_matrix[:, None, None]
        QK = softmax_temp * QK
        
        # position encoding
        QK = self.pos_encoding_layer(QK=QK, pos=pos)
        
        # Compute the attention and the weighted average
        A = self.dropout(torch.softmax(QK, dim=-1))
        V = torch.einsum("nhls,nshd->nlhd", A, values)
        
        # tmp = 0
        # for tmp_m in pos:
        #     tmp = max(tmp, len(tmp_m))
        
        # # print('full_attention_spatial | queries:{} | keys:{} | values:{} | QK:{} | A:{} | pos:{} | max(pos[i]):{} | padding_pos:{}'.format(queries.size(), keys.size(), values.size(), QK.size(), A.size(), len(pos), 0, padding_pos.size()))
        # full_attention_spatial | queries:torch.Size([150, 92, 12, 64]) | keys:torch.Size([150, 92, 12, 64]) | values:torch.Size([150, 92, 12, 64]) | QK:torch.Size([150, 12, 92, 92]) | A:torch.Size([150, 12, 92, 92]) | pos:150 | max(pos[i]):92
        # full_attention_spatial | queries:torch.Size([50, 201, 12, 64]) | keys:torch.Size([50, 201, 12, 64]) | values:torch.Size([50, 201, 12, 64]) | QK:torch.Size([50, 12, 201, 201]) | A:torch.Size([50, 12, 201, 201])
        # Let the world know of the attention matrix
        self.event_dispatcher.dispatch(AttentionEvent(self, A))

        # Make sure that what we return is contiguous
        return V.contiguous()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "full_attention_spatial", FullAttention_Spatial,
    [
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, "")),
        # ("pos_encoding_layer", Optional(GraphPositionalEncoding)),
        ("num_heads", Optional(Int))
    ]
)

class SmilesAttentionLayer(AttentionLayer):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, event_dispatcher=""):
        super(SmilesAttentionLayer, self).__init__(attention,d_model, n_heads, d_keys=d_keys,
                 d_values=d_values, event_dispatcher=event_dispatcher)
        
        self.rotaryemb = RotaryEmbedding(d_keys)
        # self.smiles_pos_encoding = SmilesPositionalEncoding(d_model=d_model//n_heads)
        # self.rotaryemb = RotaryEmbedding(d_keys)
        print('Using SmilesAttentionLayer')

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths, pos):
        # Extract the dimensions into local variables
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        
        # # position encoding
        # queries = self.smiles_pos_encoding(x=queries)
        # keys = self.smiles_pos_encoding(x=keys)
        cos, sin = self.rotaryemb(queries)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        
        values = self.value_projection(values).view(N, S, H, -1)
        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))


        # Compute the attention
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths,
            # pos
        ).view(N, L, -1)

        # Project the output and return
        return self.out_projection(new_values)


class TwoDimensionAttentionLayer(AttentionLayer):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, event_dispatcher=""):
        super(TwoDimensionAttentionLayer, self).__init__(attention,d_model, n_heads, d_keys=d_keys,
                 d_values=d_values, event_dispatcher=event_dispatcher)
        self.graph_pos_encoding = GraphPositionalEncoding(n_heads=n_heads, d_model=d_model, pos_type='laplacian')
        # self.smiles_pos_encoding = SmilesPositionalEncoding(d_model=d_model//n_heads)
        self.rotaryemb = RotaryEmbedding(d_keys)
        print('Using TwoDimensionAttentionLayer')

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths, pos=None):
        # print('In TwoDimensionAttentionLayer')
        # Extract the dimensions into local variables
        # # print(queries.shape)
        N, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project the queries/keys/values
        queries = self.query_projection(queries).view(N, L, H, -1)
        keys = self.key_projection(keys).view(N, S, H, -1)
        values = self.value_projection(values).view(N, S, H, -1)
        
        # position encoding
        # queries = self.smiles_pos_encoding(x=queries)
        # keys = self.smiles_pos_encoding(x=keys)
        # print('smiles_pos_encoding in layer')
        # pos_encoding = self.graph_pos_encoding(pos=pos, device=queries.device)
        # queries = queries + pos_encoding
        # keys = keys + pos_encoding
        # print('graph_pos_encoding in layer')
        cos, sin = self.rotaryemb(queries)
        queries, keys = apply_rotary_pos_emb(queries, keys, cos, sin)
        
        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))
                
        # pos = torch.tensor(pos)
        # Compute the attention
        # # print('self.inner_attention', self.inner_attention)
        new_values = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            query_lengths,
            key_lengths,
            pos
        ).view(N, L, -1)
        
        # Project the output and return
        # print('Out TwoDimensionAttentionLayer')
        return self.out_projection(new_values)
        

# class ThreeDimensionAttentionLayer(AttentionLayer):
#     def __init__(self, attention, d_model, n_heads, d_keys=None,
#                  d_values=None, event_dispatcher=""):
#         super(ThreeDimensionAttentionLayer, self).__init__(attention,d_model, n_heads, d_keys=d_keys,
#                  d_values=d_values, event_dispatcher=event_dispatcher)

#         # print('Using ThreeDimensionAttentionLayer')

#     def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths, pos):
#         # Extract the dimensions into local variables
#         N, L, _ = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads

#         # Project the queries/keys/values
#         queries = self.query_projection(queries).view(N, L, H, -1)
#         keys = self.key_projection(keys).view(N, S, H, -1)
#         values = self.value_projection(values).view(N, S, H, -1)
#         # Let the world know of the qkv
#         self.event_dispatcher.dispatch(QKVEvent(self, queries, keys, values))
        
#         pos = torch.tensor(pos)
#         # Compute the attention
#         new_values = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask,
#             query_lengths,
#             key_lengths,
#             pos
#         ).view(N, L, -1)
        
#         # Project the output and return
#         return self.out_projection(new_values)

#     def spatial_distance_between_nodes(self, batch_smiles):
#         # Create a list to store the RDKit molecule objects
#         mol_list = []

#         # Convert each SMILES string to an RDKit molecule object
#         for smiles in batch_smiles:
#             mol = Chem.MolFromSmiles(smiles)
#             mol = Chem.RemoveHs(mol)  # Remove hydrogen atoms if desired
#             mol_list.append(mol)

#         # Get the maximum number of atoms among the molecules in the batch
#         max_atoms = max(mol.GetNumAtoms() for mol in mol_list)

#         # Create a tensor to store the atomic coordinates for each molecule
#         coords_tensor = torch.zeros(len(mol_list), max_atoms, 3)

#         # Generate 3D coordinate representations for each molecule
#         for i, mol in enumerate(mol_list):
#             AllChem.EmbedMolecule(mol)
#             AllChem.UFFOptimizeMolecule(mol)

#             # Get the atomic coordinates for the molecule
#             coords = mol.GetConformer().GetPositions()

#             # Assign the coordinates to the tensor
#             num_atoms = mol.GetNumAtoms()
#             coords_tensor[i, :num_atoms] = torch.tensor(coords)

#         # Reshape the tensor for efficient computation
#         coords_tensor = coords_tensor.unsqueeze(1)  # Add a singleton dimension for broadcasting

#         # Calculate pairwise distances using torch
#         diff = coords_tensor - coords_tensor.transpose(2, 1)
#         distances = torch.norm(diff, dim=-1)

#         return distances

class FullAttention_Cross(Module):
    """Implement the scaled dot product attention with softmax.

    Arguments
    ---------
        softmax_temp: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, softmax_temp=None, attention_dropout=0.1,
                 event_dispatcher="", num_heads=None):
        super(FullAttention_Cross, self).__init__()
        print('Using FullAttention_Cross')
        self.softmax_temp = softmax_temp
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        # pos_encoding_layer = GraphPositionalEncoding
        # self.pos_encoding_layer = pos_encoding_layer(n_heads=num_heads)

    def forward(self, queries1, keys1, values1, queries2, keys2, values2, 
                attn_mask1, query_lengths1, key_lengths1, pos1,
                attn_mask2, query_lengths2, key_lengths2, pos2):
        """Implements the multihead softmax attention.

        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # Extract some shapes and compute the temperature
        N1, L1, H1, E1 = queries1.shape
        _, S1, _, D1 = values1.shape
        softmax_temp1 = self.softmax_temp or 1./sqrt(E1)
        N2, L2, H2, E2 = queries2.shape
        _, S2, _, D2 = values2.shape
        softmax_temp2 = self.softmax_temp or 1./sqrt(E2)

        # # pos2 encoding
        # # QK.shape = (N, H, L, S), L=S
        # padding_pos2 = torch.zeros((N, L, S))
        # for i, one_pos in enumerate(pos):
        #     padding_length = Q.shape[-1] - one_pos.shape[-1]
        #     padded_one_pos = torch.nn.functional.pad(one_pos, (0, padding_length, 0, padding_length))
        #     padding_pos2[i, :, :] = padded_one_pos
        # padding_pos2 = self.spatial_pos_encoder(padding_pos2).permute(0, 3, 1, 2) # (N, L, S) -> (N, L, S, H) -> (N, H, L, S)
        # queries2 = queries2 + padding_pos2
        # keys2 = keys2 + padding_pos2
        
#         cos, sin = self.rotaryemb(queries1)
#         queries1, keys1 = apply_rotary_pos_emb(queries1, keys1, cos, sin)
#         cos, sin = self.rotaryemb(queries2)
#         queries2, keys2 = apply_rotary_pos_emb(queries2, keys2, cos, sin)
        
        # Compute the unnormalized attention and apply the masks
        QK1 = torch.einsum("nlhe,nshe->nhls", queries1, keys2) # nl1h1e1, nl2h2e2 -> nhl1l2, h1=h2, e1=e2
        if not attn_mask2.all_ones:
            QK1 = QK1 + attn_mask2.additive_matrix
        QK1 = QK1 + key_lengths2.additive_matrix[:, None, None]
        # Compute the attention and the weighted average
        # print('QK1', QK1.shape)
        # print('pos1', pos1.shape)
        A1 = softmax_temp2 * QK1# + pos1
        A1 = self.dropout(torch.softmax(A1, dim=-1))
        V1 = torch.einsum("nhls,nshd->nlhd", A1, values2) # nhl1l2, nl2hd -> nhl1l2
        
        # Compute the unnormalized attention and apply the masks
        QK2 = torch.einsum("nlhe,nshe->nhls", queries2, keys1) # nl2he, nl1he -> nhl2l1
        if not attn_mask1.all_ones:
            QK2 = QK2 + attn_mask1.additive_matrix
        QK2 = QK2 + key_lengths1.additive_matrix[:, None, None]
        # Compute the attention and the weighted average
        # print('QK2', QK2.shape, (softmax_temp1 * QK2)[0, 0,:,:])
        # print('pos2', pos2.shape, pos2[0,0,:,:])
        A2 = softmax_temp1 * QK2# + pos2
        # print('no pos1 #and pos2') 
        A2 = self.dropout(torch.softmax(A2, dim=-1))
        V2 = torch.einsum("nhls,nshd->nlhd", A2, values1) # nhl1l2, nl2hd -> nhl1l2

        # Let the world know of the attention matrix
        self.event_dispatcher.dispatch(AttentionEvent(self, A1))
        self.event_dispatcher.dispatch(AttentionEvent(self, A2))

        # Make sure that what we return is contiguous
        return V1.contiguous(), V2.contiguous()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "full_attention_cross", FullAttention_Cross,
    [
        ("softmax_temp", Optional(Float)),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, "")),
        ("num_heads", Optional(Int))
    ]
)


class CrossAttentionLayer(AttentionLayer):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, event_dispatcher=""):
        super(CrossAttentionLayer, self).__init__(attention, d_model, n_heads, d_keys=d_keys,
                 d_values=d_values, event_dispatcher=event_dispatcher)
        
        self.query_projection = None
        self.key_projection = None
        self.value_projection = None
        self.out_projection = None
        self.out_projection1 = Linear(d_values * n_heads, d_model)
        self.out_projection2 = Linear(d_values * n_heads, d_model)
        self.query_projection1 = Linear(d_model, d_keys * n_heads)
        self.query_projection2 = Linear(d_model, d_keys * n_heads)
        self.key_projection1 = Linear(d_model, d_keys * n_heads)
        self.key_projection2 = Linear(d_model, d_keys * n_heads)
        self.value_projection1 = Linear(d_model, d_values * n_heads)
        self.value_projection2 = Linear(d_model, d_values * n_heads)

        self.rotaryemb = RotaryEmbedding(d_keys)
        print('Using CrossAttentionLayer')
   #      self.inner_attention = attention
   #      self.query_projection = Linear(d_model, d_keys * n_heads)
   #      self.key_projection = Linear(d_model, d_keys * n_heads)
   #      self.value_projection = Linear(d_model, d_values * n_heads)
   #      self.out_projection = Linear(d_values * n_heads, d_model)
   #      self.n_heads = n_heads
   #      self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        
        
        
    def forward(self, queries1, keys1, values1, queries2, keys2, values2, 
                attn_mask1, query_lengths1, key_lengths1, pos1, 
                attn_mask2, query_lengths2, key_lengths2, pos2):
        # # print('--CrossAttentionLayer')
        # Extract the dimensions into local variables
        # print('queries1', queries1.shape)
        N1, L1, _ = queries1.shape
        _, S1, _ = keys1.shape
        N2, L2, _ = queries2.shape
        _, S2, _ = keys2.shape
        H = self.n_heads

        # # print('queries1:{} |  keys1:{} |  values1:{} |  queries2:{} |  keys2:{} |  values2:{} |  \
        #         attn_mask1:{} |  query_lengths1:{} |  key_lengths1:{} |  pos1:{} |  \
        #         attn_mask2:{} |  query_lengths2:{} |  key_lengths2:{} |  pos2:{}'.format(
        #         queries1.shape, keys1.shape, values1.shape, queries2.shape, keys2.shape, values2.shape, 
        #         attn_mask1.shape, query_lengths1.shape, key_lengths1.shape, pos1.shape, 
        #         attn_mask2.shape, query_lengths2.shape, key_lengths2.shape, pos2.shape))
        # queries1:torch.Size([8, 50, 768]) |  keys1:torch.Size([8, 50, 768]) |  values1:torch.Size([8, 50, 768]) |  queries2:torch.Size([8, 30, 768]) |  keys2:torch.Size([8, 30, 768]) |  values2:torch.Size([8, 30, 768]) |                  attn_mask1:torch.Size([50, 50]) |  query_lengths1:torch.Size([8, 50]) |  key_lengths1:torch.Size([8, 50]) |  pos1:torch.Size([8, 50, 50]) |                  attn_mask2:torch.Size([30, 30]) |  query_lengths2:torch.Size([8, 30]) |  key_lengths2:torch.Size([8, 30]) |  pos2:torch.Size([8, 30, 30])
        # Project the queries/keys/values
        queries1 = self.query_projection1(queries1).view(N1, L1, H, -1)
        keys1 = self.key_projection1(keys1).view(N1, S1, H, -1)
        values1 = self.value_projection1(values1).view(N1, S1, H, -1)

        # Project the queries/keys/values
        queries2 = self.query_projection2(queries2).view(N2, L2, H, -1)
        keys2 = self.key_projection2(keys2).view(N2, S2, H, -1)
        values2 = self.value_projection2(values2).view(N2, S2, H, -1)
        
        cos, sin = self.rotaryemb(queries1)
        queries1, keys1 = apply_rotary_pos_emb(queries1, keys1, cos, sin)
        cos, sin = self.rotaryemb(queries2)
        queries2, keys2 = apply_rotary_pos_emb(queries2, keys2, cos, sin)
        # print('query2 no rotary embedding')
        
        # Let the world know of the qkv
        self.event_dispatcher.dispatch(QKVEvent(self, queries1, keys1, values1))
        self.event_dispatcher.dispatch(QKVEvent(self, queries2, keys2, values2))
        
        # pos = torch.tensor(pos)
        # Compute the attention
        # # print('self.inner_attention', self.inner_attention)
        new_values1, new_values2 = self.inner_attention(
            queries1,
            keys1,
            values1,
            queries2,
            keys2,
            values2,
            attn_mask1,
            query_lengths1,
            key_lengths1,
            pos1,
            attn_mask2,
            query_lengths2,
            key_lengths2,
            pos2
        )
        new_values1 = new_values1.view(N1, L1, -1)
        new_values2 = new_values2.view(N2, L2, -1)
        
        # Project the output and return
        return self.out_projection1(new_values1), self.out_projection2(new_values2)
