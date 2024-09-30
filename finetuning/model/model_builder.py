from torch.nn import LayerNorm

from .attention_layer import SmilesAttentionLayer, TwoDimensionAttentionLayer, CrossAttentionLayer, CrossAttentionLinearRotaryLayer
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer
from fast_transformers.builders.base import BaseBuilder
from fast_transformers.builders.transformer_builders import BaseTransformerEncoderBuilder
from fast_transformers.builders.attention_builders import AttentionBuilder

import torch
from torch.nn import Dropout, LayerNorm, Linear, Module, ModuleList
import torch.nn.functional as F

from fast_transformers.events.event_dispatcher import EventDispatcher
from fast_transformers.masking import FullMask, LengthMask
from utils import merge_one_matrix
from collections import defaultdict
import sys
from .attention_layer import RotateAttentionLayer 

HYPERGRAPH_TYPE = 'att' # att | avg
N_HEAD = 8 #12#8
N_EMB = 128 #64#128

class RotateEncoderBuilder(BaseTransformerEncoderBuilder):
    """Build a batch transformer encoder with Relative Rotary embeddings
    for training or processing of sequences all elements at a time.

    Example usage:

        builder = RotateEncoderBuilder()
        builder.n_layers = 12
        builder.n_heads = 8
        builder.feed_forward_dimensions = 1024
        builder.query_dimensions = 64
        builder.value_dimensions = 64
        builder.dropout = 0.1
        builder.attention_dropout = 0.1
        builder.attention_type = "linear"
        transformer = builder.get()
    """
    def _get_attention_builder(self):
        """Return an instance of the appropriate attention builder."""
        return AttentionBuilder()

    def _get_attention_layer_class(self):
        """Return the class for the layer that projects queries keys and
        values."""
        return RotateAttentionLayer

    def _get_encoder_class(self):
        """Return the class for the transformer encoder."""
        return TransformerEncoder

    def _get_encoder_layer_class(self):
        """Return the class for the transformer encoder layer."""
        return TransformerEncoderLayer

    
class CrossAttentionLinearRotary(BaseTransformerEncoderBuilder):
    def _get_attention_builder(self):
        return AttentionBuilder()

    def _get_attention_layer_class(self):
        return CrossAttentionLinearRotaryLayer

    def _get_encoder_class(self):
        return TransformerEncoderCrossAttentionLinearRotary

    def _get_encoder_layer_class(self):
        return TransformerEncoderLayerCrossAttentionLinearRotary
    

class TransformerEncoderCrossAttentionLinearRotary(Module):
    def __init__(self, layers, norm_layer=None, event_dispatcher=""):
        super(TransformerEncoderCrossAttentionLinearRotary, self).__init__()
        print('Using TransformerEncoderCrossAttention')
        self.layers = ModuleList(layers)
        self.norm1 = norm_layer
        self.norm2 = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x1, x2, attn_mask1=None, length_mask1=None, pos1=None, attn_mask2=None, length_mask2=None, pos2=None, hyper_adj2=None):
        # Normalize the masks
        N = x1.shape[0]
        L = x1.shape[1]
        attn_mask1 = attn_mask1 or FullMask(L, device=x1.device)
        length_mask1 = length_mask1 or \
            LengthMask(x1.new_full((N,), L, dtype=torch.int64))

        N = x2.shape[0]
        L = x2.shape[1]
        attn_mask2 = attn_mask2 or FullMask(L, device=x2.device)
        length_mask2 = length_mask2 or \
            LengthMask(x2.new_full((N,), L, dtype=torch.int64))

        # Apply all the transformers
        for layer in self.layers:
            x1, x2 = layer(x1, x2, attn_mask1=attn_mask1, length_mask1=length_mask1, pos1=pos1, 
            attn_mask2=attn_mask2, length_mask2=length_mask2, pos2=pos2, hyper_adj2=hyper_adj2)

        # Apply the normalization if needed
        if self.norm1 is not None:
            x1 = self.norm1(x1)
        if self.norm2 is not None:
            x2 = self.norm2(x2)

        return x1, x2


class TransformerEncoderLayerCrossAttentionLinearRotary(Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(TransformerEncoderLayerCrossAttentionLinearRotary, self).__init__()
        print('Using TransformerEncoderLayerCrossAttention')
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm11 = LayerNorm(d_model)
        self.norm12 = LayerNorm(d_model)
        self.norm21 = LayerNorm(d_model)
        self.norm22 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        
    def forward(self, x1, x2, attn_mask1=None, length_mask1=None, pos1=None, attn_mask2=None, length_mask2=None, pos2=None, hyper_adj2=None):
        # Normalize the masks
        N = x1.shape[0]
        L = x1.shape[1]
        attn_mask1 = attn_mask1 or FullMask(L, device=x1.device)
        length_mask1 = length_mask1 or \
            LengthMask(x1.new_full((N,), L, dtype=torch.int64))

        N = x2.shape[0]
        L = x2.shape[1]
        attn_mask2 = attn_mask2 or FullMask(L, device=x2.device)
        length_mask2 = length_mask2 or \
            LengthMask(x2.new_full((N,), L, dtype=torch.int64))
        
        x1, x2 = self.attention(
            x1, x1, x1,
            x2, x2, x2,
            attn_mask1=attn_mask1,
            query_lengths1=length_mask1,
            key_lengths1=length_mask1,
            pos1=pos1,
            attn_mask2=attn_mask2,
            query_lengths2=length_mask2,
            key_lengths2=length_mask2,
            pos2=pos2,
        )
        
        x1 = x1 + self.dropout(x1)
        x2 = x2 + self.dropout(x2)
        
        # Run the fully connected part of the layer
        y1 = x1 = self.norm11(x1)
        y1 = self.dropout(self.activation(self.linear1(y1)))
        y1 = self.dropout(self.linear2(y1))
        
        y2 = x2 = self.norm21(x2)
        y2 = self.dropout(self.activation(self.linear1(y2)))
        y2 = self.dropout(self.linear2(y2))

        return self.norm12(x1+y1), self.norm22(x2+y2)
        
        

        
        
        
# -----------------------

class HypergraphLayer(Module):
    def __init__(self, d_model, n_head=N_HEAD):
        super(HypergraphLayer, self).__init__()
        print('Using HypergraphLayer')
        
        self.n_head = n_head
        self.n_emb = N_EMB
        self.activation = F.gelu
        
        if HYPERGRAPH_TYPE == 'avg':
            print('Using hyperGraph with Average pooling')
            self.hyper_layer = self.hyperNet_vHE 
            self.eps = 1e-20
            self.hyper_linear_avg1 = Linear(d_model*2, d_model)
            self.hyper_linear_avg2 = Linear(d_model*2, d_model)
            print('Using hyperGraph with Attention pooling')
        elif HYPERGRAPH_TYPE == 'att':
            self.hyper_layer = self.hyperAttNet_vHE 
            self.eps2 = -1e20
            self.hyper_linear1 = Linear(d_model//n_head, d_model//n_head)
            self.hyper_linear2 = Linear(d_model//n_head, 1)
            self.hyper_linear3 = Linear(d_model//n_head, d_model//n_head)
            self.hyper_linear4 = Linear(d_model//n_head*2, 1)
        else:
            raise ValueError('Error HYPERGRAPH_TYPE:{}'.format(HYPERGRAPH_TYPE))
    
    def forward(self, x, hyper_adj):
        return self.hyper_layer(x, hyper_adj)
    
    def hyperNet_vHE(self, x, hyper_adj):
        '''
        vHE: virtual hyper edge
        hyper_adj: shape:(batch_num, node, hyper_edge) (row:nodes, column: hyper_edge)
        '''
        # hyper_adj = merge_one_matrix(hyper_adj)
        columns_sum = hyper_adj.sum(dim=1, keepdim=True) + self.eps
        hyper_adj_cnorm = hyper_adj / columns_sum
        
        # step1: hyper-edge = att(node_in)
        hyper_edge = torch.einsum("npl,nle->npe", hyper_adj_cnorm.permute(0, 2, 1), x) # hyper_adj_norm.T: row:edge, column:node
        
        # step2: node' = att(hyper-edge)
        rows_sum = hyper_adj.sum(dim=2, keepdim=True) + self.eps
        hyper_adj_rnorm = hyper_adj / rows_sum
        x_new = torch.einsum("nlp,npe->nle", hyper_adj_rnorm, hyper_edge) 
        
        # step3: node_out = node_in + node'
        return self.activation(self.hyper_linear_avg1(torch.cat((x, x_new), dim=-1)))
    
    def hyperNet_trueNode(self, x, hyper_adj):
        '''
        Current BUGs: can not cat x_new with hyper_edge_new, due to the unertain shape...
        hyper_adj: shape:(batch_num, node, hyper_edge) (row:nodes, column: hyper_edge)
        '''
        pass
#         _, L1, _ = hyper_adj.shape
#         _, L2, _ = x.shape
#         hyper_adj = F.pad(hyper_adj, (0, L2-L1, 0, 0, 0, 0), mode='constant', value=0) # (N, L2, P)
        
#         columns_sum = hyper_adj.sum(dim=1, keepdim=True) + self.eps
#         hyper_adj_cnorm = hyper_adj / columns_sum
        
#         # step1: hyper-edge = AVG(node_in)
#         hyper_edge = torch.einsum("npl,nle->npe", hyper_adj_cnorm.T, x) # hyper_adj_norm.T: row:edge, column:node
#         hyper_edge = self.activation(self.hyper_linear1(torch.cat((x[:,-P:,:], hyper_edge), dim=-1))) # (N, P, E)
        
#         # step2: node' = AVG(hyper-edge)
#         rows_sum = hyper_adj.sum(dim=2, keepdim=True) + self.eps
#         hyper_adj_rnorm = hyper_adj / rows_sum
#         x_new = torch.einsum("nlp,npe->nle", hyper_adj_rnorm, hyper_edge) # (N, L1, E) hyper_adj_norm.T: row:edge, column:node
        
#         # step3: node_out = node_in + node'
#         return self.activation(self.hyper_linear2(torch.cat((x, x_new), dim=-1)))
        
    def hyperAttNet_vHE(self, input_x, input_hyper_adj):
        # print('input_hyper_adj', input_hyper_adj.shape)
        # hyper_adj = merge_one_matrix(hyper_adj)
        hyper_adj = input_hyper_adj.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        N, _, L, P = hyper_adj.shape
        # print('input_x', input_x.shape)
        x = input_x.view(N, L, self.n_head, self.n_emb).permute(0, 2, 1, 3)
        
        x = self.hyper_linear1(x) # nhle
        # node to hyper_edge
        x_value = self.hyper_linear2(self.activation(x)).repeat(1, 1, 1, P) # (N, H, L, 1)
        x_value[hyper_adj==0.0] = self.eps2 # self.eps2 # do not calculate zero in softmax
        att_score_edge = F.softmax(x_value, dim=1) # (N, H, L, P)
        # print('att_score_edge.T, att_score_edge, x', att_score_edge.T.shape, att_score_edge.shape, x.shape)
        hyper_edge = self.activation(torch.einsum("nhpl,nhle->nhpe", att_score_edge.permute(0, 1, 3, 2), x)) # hyper_adj_norm: row:node, column:edge
        
        # hyper_edge to node
        hyper_edge = self.hyper_linear3(hyper_edge) # (nhpe)
        hyper_edge_value = hyper_edge.unsqueeze(3).repeat(1,1,1,L,1) # nhpe -> nhp1e -> nhple
        x = x.unsqueeze(2).repeat(1,1,P,1,1) # nhle -> nh1le -> nhple
        # print('hyper_edge, x', hyper_edge.shape, x.shape)
        e_values = self.hyper_linear4(
            self.activation(
                torch.cat(
                    (hyper_edge_value, x), dim=-1 # nple -> npl(2e)
                )
            )
        ).squeeze(-1).permute(0,1,3,2)  # nhple -> nhpl(2e) -> nhpl1 -> nhpl -> nhlp
        # print('hyper_edge_att:{hyper_edge_att} | x:{x} | e_values:{e_values}'.format(hyper_edge_att=hyper_edge_att.shape, x=x.shape, e_values=e_values.shape))
        e_values[hyper_adj==0.0] = self.eps2 # self.eps2 # do not calculate zero in softmax
        att_score_node = F.softmax(e_values, dim=1) # (N, L, P)
        x_nodes = torch.einsum("nhlp,nhpe->nhle", att_score_node, hyper_edge)
        
        output_x = self.activation(x_nodes).permute(0,2,1,3).contiguous().view(N,L,-1)
        
        output_x = output_x.view(-1,output_x.shape[-1])
        input_x = input_x.view(-1, input_x.shape[-1])
        hyperadj_exist = torch.nonzero(torch.sum(input_hyper_adj, dim=-1).view(-1)==0.0).view(-1)
        # print('hyperadj_exist',hyperadj_exist, hyperadj_exist.shape)
        output_x[hyperadj_exist] = input_x[hyperadj_exist]
        
        return output_x.view(N,L,-1)
    
class TransformerEncoderPos(Module):
    """TransformerEncoder is little more than a sequence of transformer encoder
    layers.

    It contains an optional final normalization layer as well as the ability to
    create the masks once and save some computation.

    Arguments
    ---------
        layers: list, TransformerEncoderLayer instances or instances that
                implement the same interface.
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, layers, norm_layer=None, event_dispatcher=""):
        super(TransformerEncoderPos, self).__init__()
        print('Using TransformerEncoderPos')
        self.layers = ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, attn_mask=None, length_mask=None, pos=None):
        """Apply all transformer encoder layers to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor of each transformer encoder layer.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
            
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Apply all the transformers
        for layer in self.layers:
            # print('TransformerEncoderPos x.shape', x.shape)
            x = layer(x, attn_mask=attn_mask, length_mask=length_mask, pos=pos)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        return x

class TransformerEncoderLayerPos(Module):
    """Self attention and feed forward network with skip connections.
    
    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(TransformerEncoderLayerPos, self).__init__()
        print('Using TransformerEncoderLayerPos')
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, attn_mask=None, length_mask=None, pos=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Run self attention and add it to the input
        # print('TransformerEncoderLayerPos x.shape', x.shape)
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask,
            pos=pos
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        return self.norm2(x+y)

class TransformerEncoderHyperGraph(Module):
    """TransformerEncoder is little more than a sequence of transformer encoder
    layers.

    It contains an optional final normalization layer as well as the ability to
    create the masks once and save some computation.

    Arguments
    ---------
        layers: list, TransformerEncoderLayer instances or instances that
                implement the same interface.
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, layers, norm_layer=None, event_dispatcher=""):
        super(TransformerEncoderHyperGraph, self).__init__()
        print('Using TransformerEncoderHyperGraph')
        self.layers = ModuleList(layers)
        self.norm = norm_layer
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x, attn_mask=None, length_mask=None, pos=None, hyper_adj=None):
        """Apply all transformer encoder layers to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor of each transformer encoder layer.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
            
        """
        # print('In TransformerEncoderHyperGraph')
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # Apply all the transformers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, length_mask=length_mask, pos=pos, hyper_adj=hyper_adj)

        # Apply the normalization if needed
        if self.norm is not None:
            x = self.norm(x)

        # print('Out TransformerEncoderHyperGraph')
        return x
    
class TransformerEncoderLayerHypergraph(Module):
    """Self attention and feed forward network with skip connections.
    
    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher="", hyper_mode=None):
        super(TransformerEncoderLayerHypergraph, self).__init__()
        print('Using TransformerEncoderLayerHypergraph')
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

        self.hyper_layer = HypergraphLayer(d_model)
        
    def forward(self, x, attn_mask=None, length_mask=None, pos=None, hyper_adj=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # print('In TransformerEncoderLayerHypergraph')
        # Normalize the masks
        N = x.shape[0]
        L = x.shape[1]
        attn_mask = attn_mask or FullMask(L, device=x.device)
        length_mask = length_mask or \
            LengthMask(x.new_full((N,), L, dtype=torch.int64))

        # hyper-graph learning
        # ori_x = x
        x = self.hyper_layer(x, hyper_adj)
        # print('No hyper_layer')
        
        # Run self attention and add it to the input
        x = x + self.dropout(self.attention(
            x, x, x,
            attn_mask=attn_mask,
            query_lengths=length_mask,
            key_lengths=length_mask,
            pos=pos
        ))

        # Run the fully connected part of the layer
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.linear1(y)))
        y = self.dropout(self.linear2(y))

        # print('Out TransformerEncoderLayerHypergraph')
        return self.norm2(x+y)
    
    def hyperNet_vHE(self, x, hyper_adj):
        '''
        vHE: virtual hyper edge
        hyper_adj: shape:(batch_num, node, hyper_edge) (row:nodes, column: hyper_edge)
        '''
        # hyper_adj = merge_one_matrix(hyper_adj)
        print('hyper_adj', hyper_adj)
        N, H, L, _ = x.view(N, L, self.n_head, -1).permute(0, 2, 1, 3).shape
        
        # for n in range(N):
        #     for 
            
        
        # for n in range(N): # graph
        #     for h in range(self.n_head):
        
    
    def hyperNet_trueNode(self, x, hyper_adj):
        '''
        Current BUGs: can not cat x_new with hyper_edge_new, due to the unertain shape...
        hyper_adj: shape:(batch_num, node, hyper_edge) (row:nodes, column: hyper_edge)
        '''
        pass
#         _, L1, _ = hyper_adj.shape
#         _, L2, _ = x.shape
#         hyper_adj = F.pad(hyper_adj, (0, L2-L1, 0, 0, 0, 0), mode='constant', value=0) # (N, L2, P)
        
#         columns_sum = hyper_adj.sum(dim=1, keepdim=True) + self.eps
#         hyper_adj_cnorm = hyper_adj / columns_sum
        
#         # step1: hyper-edge = AVG(node_in)
#         hyper_edge = torch.einsum("npl,nle->npe", hyper_adj_cnorm.T, x) # hyper_adj_norm.T: row:edge, column:node
#         hyper_edge = self.activation(self.hyper_linear1(torch.cat((x[:,-P:,:], hyper_edge), dim=-1))) # (N, P, E)
        
#         # step2: node' = AVG(hyper-edge)
#         rows_sum = hyper_adj.sum(dim=2, keepdim=True) + self.eps
#         hyper_adj_rnorm = hyper_adj / rows_sum
#         x_new = torch.einsum("nlp,npe->nle", hyper_adj_rnorm, hyper_edge) # (N, L1, E) hyper_adj_norm.T: row:edge, column:node
        
#         # step3: node_out = node_in + node'
#         return self.activation(self.hyper_linear2(torch.cat((x, x_new), dim=-1)))
        
    def hyperAttNet_vHE(self, input_x, input_hyper_adj):
        # hyper_adj = merge_one_matrix(hyper_adj)
        hyper_adj = input_hyper_adj.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        N, _, L, P = hyper_adj.shape
        x = input_x.view(N, L, self.n_head, -1).permute(0, 2, 1, 3)
        
        x = self.hyper_linear1(x) # nhle
        # node to hyper_edge
        x_value = self.hyper_linear2(self.activation(x)).repeat(1, 1, 1, P) # (N, H, L, 1)
        x_value[hyper_adj==0.0] = self.eps2 # self.eps2 # do not calculate zero in softmax
        att_score_edge = F.softmax(x_value, dim=1) # (N, H, L, P)
        # print('att_score_edge.T, att_score_edge, x', att_score_edge.T.shape, att_score_edge.shape, x.shape)
        hyper_edge = self.activation(torch.einsum("nhpl,nhle->nhpe", att_score_edge.permute(0, 1, 3, 2), x)) # hyper_adj_norm: row:node, column:edge
        
        # hyper_edge to node
        hyper_edge = self.hyper_linear3(hyper_edge) # (nhpe)
        hyper_edge_value = hyper_edge.unsqueeze(3).repeat(1,1,1,L,1) # nhpe -> nhp1e -> nhple
        x = x.unsqueeze(2).repeat(1,1,P,1,1) # nhle -> nh1le -> nhple
        # print('hyper_edge, x', hyper_edge.shape, x.shape)
        e_values = self.hyper_linear4(
            self.activation(
                torch.cat(
                    (hyper_edge_value, x), dim=-1 # nple -> npl(2e)
                )
            )
        ).squeeze(-1).permute(0,1,3,2)  # nhple -> nhpl(2e) -> nhpl1 -> nhpl -> nhlp
        # print('hyper_edge_att:{hyper_edge_att} | x:{x} | e_values:{e_values}'.format(hyper_edge_att=hyper_edge_att.shape, x=x.shape, e_values=e_values.shape))
        e_values[hyper_adj==0.0] = self.eps2 # self.eps2 # do not calculate zero in softmax
        att_score_node = F.softmax(e_values, dim=1) # (N, L, P)
        x_nodes = torch.einsum("nhlp,nhpe->nhle", att_score_node, hyper_edge)
        
        output_x = self.activation(x_nodes).permute(0,2,1,3).contiguous().view(N,L,-1)
        
        output_x = output_x.view(-1,output_x.shape[-1])
        input_x = input_x.view(-1, input_x.shape[-1])
        hyperadj_exist = torch.nonzero(torch.sum(input_hyper_adj, dim=-1).view(-1)==0.0).view(-1)
        # print('hyperadj_exist',hyperadj_exist, hyperadj_exist.shape)
        output_x[hyperadj_exist] = input_x[hyperadj_exist]
        
        return output_x.view(N,L,-1)
        
    def hyperAttNet_vHE2(self, x, hyper_adj):
        '''
        vHE: virtual hyper edge
        mask att with hyper_adj before softmax
        '''
        # print('In hyperAttNet_vHE')
        # hyper_adj = merge_one_matrix(hyper_adj)
        N, L, _ = x.shape
        x = x.view(N, L, self.n_head, -1).permute(0, 2, 1, 3)
        batch_num, num_heads, num_nodes, _ = x.shape
        
        for b in range(batch_num):
            hyperedge_to_node, _ = hyper_adj[b]
            node_to_hyperedge = defaultdict(list)
            for h in range(num_heads):
                # Node to Hyperedge Attention
                hyperedge_emb = []
                for key, he in hyperedge_to_node.items():
                    for node in he:
                        node_to_hyperedge[node].append(len(hyperedge_emb))
                    # print('he', he)
                    node_subset = x[b, h, he, :]
                    attention_weights = F.softmax(torch.sum(node_subset, dim=0, keepdim=True), dim=0)
                    hyperedge_emb.append(torch.sum(attention_weights * node_subset, dim=0))
                hyperedge_emb = torch.stack(hyperedge_emb)

                # Hyperedge to Node Attention
                for key, n in node_to_hyperedge.items():
                    hyperedge_subset = hyperedge_emb[n, :]
                    attention_weights = F.softmax(torch.sum(hyperedge_subset, dim=0, keepdim=True), dim=0)
                    x[b, h, key, :] = torch.sum(attention_weights * hyperedge_subset, dim=0)

        return x.permute(0, 2, 1, 3).view(N, L, -1)
        
        # return self.activation(x_nodes)
        
        
class SmilesEncoderBuilder(BaseTransformerEncoderBuilder):
    """Build a batch transformer encoder with Relative Rotary embeddings
    for training or processing of sequences all elements at a time.

    Example usage:

        builder = SmilesEncoderBuilder()
        builder.n_layers = 12
        builder.n_heads = 8
        builder.feed_forward_dimensions = 1024
        builder.query_dimensions = 64
        builder.value_dimensions = 64
        builder.dropout = 0.1
        builder.attention_dropout = 0.1
        builder.attention_type = "linear"
        transformer = builder.get()
    """
    def _get_attention_builder(self):
        """Return an instance of the appropriate attention builder."""
        return AttentionBuilder()

    def _get_attention_layer_class(self):
        """Return the class for the layer that projects queries keys and
        values."""
        return SmilesAttentionLayer

    def _get_encoder_class(self):
        """Return the class for the transformer encoder."""
        return TransformerEncoderPos

    def _get_encoder_layer_class(self):
        """Return the class for the transformer encoder layer."""
        return TransformerEncoderLayerPos

class TwoDimensionEncoderBuilder(BaseTransformerEncoderBuilder):
    """Build a batch transformer encoder with Relative Rotary embeddings
    for training or processing of sequences all elements at a time.

    Example usage:

        builder = TwoDimensionEncoderBuilder()
        builder.n_layers = 12
        builder.n_heads = 8
        builder.feed_forward_dimensions = 1024
        builder.query_dimensions = 64
        builder.value_dimensions = 64
        builder.dropout = 0.1
        builder.attention_dropout = 0.1
        builder.attention_type = "linear"
        transformer = builder.get()
    """
    def _get_attention_builder(self):
        """Return an instance of the appropriate attention builder."""
        return AttentionBuilder()

    def _get_attention_layer_class(self):
        """Return the class for the layer that projects queries keys and
        values."""
        return TwoDimensionAttentionLayer#TwoDimensionAttentionLayer

    def _get_encoder_class(self):
        """Return the class for the transformer encoder."""
        return TransformerEncoderHyperGraph#TransformerEncoderHyperGraph

    def _get_encoder_layer_class(self):
        """Return the class for the transformer encoder layer."""
        return TransformerEncoderLayerHypergraph#TransformerEncoderLayerHypergraph

# class ThreeDimensionEncoderBuilder(BaseTransformerEncoderBuilder):
#     """Build a batch transformer encoder with Relative Rotary embeddings
#     for training or processing of sequences all elements at a time.

#     Example usage:

#         builder = ThreeDimensionEncoderBuilder()
#         builder.n_layers = 12
#         builder.n_heads = 8
#         builder.feed_forward_dimensions = 1024
#         builder.query_dimensions = 64
#         builder.value_dimensions = 64
#         builder.dropout = 0.1
#         builder.attention_dropout = 0.1
#         builder.attention_type = "linear"
#         transformer = builder.get()
#     """
#     def _get_attention_builder(self):
#         """Return an instance of the appropriate attention builder."""
#         return AttentionBuilder()

#     def _get_attention_layer_class(self):
#         """Return the class for the layer that projects queries keys and
#         values."""
#         return ThreeDimensionAttentionLayer

#     def _get_encoder_class(self):
#         """Return the class for the transformer encoder."""
#         return TransformerEncoder

#     def _get_encoder_layer_class(self):
#         """Return the class for the transformer encoder layer."""
#         return TransformerEncoderLayer


class GraphAttnBias(Module):
    """
    Compute attention bias for each head.
    """

    def __init__(
        self,
        num_heads=N_HEAD,
        num_spatial=300,
    ):
        super(GraphAttnBias, self).__init__()
        self.num_heads = num_heads

        self.smiles_pos_encoder = torch.nn.Embedding(num_spatial, num_heads, padding_idx=0)
        self.graph_pos_encoder = torch.nn.Embedding(num_spatial, num_heads, padding_idx=0)

    def forward(self, spatial_pos):
        # spatial pos
        # [n_graph, n_token, n_node, n_head] -> [n_graph, n_head, n_token, n_node]
        smiles_pos_bias = self.smiles_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
        # [n_graph, n_node, n_token, n_head] -> [n_graph, n_head, n_node, n_token]
        graph_pos_bias = self.graph_pos_encoder(spatial_pos.permute(0, 2, 1)).permute(0, 3, 1, 2)

        return smiles_pos_bias, graph_pos_bias

class TransformerEncoderCrossAttention(Module):
    """TransformerEncoder is little more than a sequence of transformer encoder
    layers.

    It contains an optional final normalization layer as well as the ability to
    create the masks once and save some computation.

    Arguments
    ---------
        layers: list, TransformerEncoderLayer instances or instances that
                implement the same interface.
        norm_layer: A normalization layer to be applied to the final output
                    (default: None which means no normalization)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, layers, norm_layer=None, event_dispatcher=""):
        super(TransformerEncoderCrossAttention, self).__init__()
        print('Using TransformerEncoderCrossAttention')
        self.layers = ModuleList(layers)
        self.norm1 = norm_layer
        self.norm2 = norm_layer
        self.spatial_bias = GraphAttnBias()
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, x1, x2, attn_mask1=None, length_mask1=None, pos1=None, attn_mask2=None, length_mask2=None, pos2=None, hyper_adj2=None):
        """Apply all transformer encoder layers to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor of each transformer encoder layer.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
            
        """
        # Normalize the masks
        N = x1.shape[0]
        L = x1.shape[1]
        attn_mask1 = attn_mask1 or FullMask(L, device=x1.device)
        length_mask1 = length_mask1 or \
            LengthMask(x1.new_full((N,), L, dtype=torch.int64))

        N = x2.shape[0]
        L = x2.shape[1]
        attn_mask2 = attn_mask2 or FullMask(L, device=x2.device)
        length_mask2 = length_mask2 or \
            LengthMask(x2.new_full((N,), L, dtype=torch.int64))

        # compute pos encoding for each head
        # print('pos2', pos2.shape)
        pos2, pos1 = self.spatial_bias(pos2)
        # print('pos1', pos1.shape)
        # print('pos2', pos2.shape)
        
        # Apply all the transformers
        for layer in self.layers:
            x1, x2 = layer(x1, x2, attn_mask1=attn_mask1, length_mask1=length_mask1, pos1=pos1, 
            attn_mask2=attn_mask2, length_mask2=length_mask2, pos2=pos2, hyper_adj2=hyper_adj2)

        # Apply the normalization if needed
        if self.norm1 is not None:
            x1 = self.norm1(x1)
        if self.norm2 is not None:
            x2 = self.norm2(x2)

        return x1, x2

class TransformerEncoderLayerCrossAttentionHyperGraph(Module):
    """Self attention and feed forward network with skip connections.
    
    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(TransformerEncoderLayerCrossAttention, self).__init__()
        print('Using TransformerEncoderLayerCrossAttention')
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        
        self.hyper_layer = HypergraphLayer(d_model)
        
    def forward(self, x1, x2, attn_mask1=None, length_mask1=None, pos1=None, attn_mask2=None, length_mask2=None, pos2=None, hyper_adj2=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x1.shape[0]
        L = x1.shape[1]
        attn_mask1 = attn_mask1 or FullMask(L, device=x1.device)
        length_mask1 = length_mask1 or \
            LengthMask(x1.new_full((N,), L, dtype=torch.int64))

        N = x2.shape[0]
        L = x2.shape[1]
        attn_mask2 = attn_mask2 or FullMask(L, device=x2.device)
        length_mask2 = length_mask2 or \
            LengthMask(x2.new_full((N,), L, dtype=torch.int64))
        
        # # hyper-graph learning
        x2 = self.hyper_layer(x2, hyper_adj2)
        # # print('no hypergraph')
        
        x1, x2 = self.attention(
            x1, x1, x1,
            x2, x2, x2,
            attn_mask1=attn_mask1,
            query_lengths1=length_mask1,
            key_lengths1=length_mask1,
            pos1=pos1,
            attn_mask2=attn_mask2,
            query_lengths2=length_mask2,
            key_lengths2=length_mask2,
            pos2=pos2,
        )
        
        x1 = x1 + self.dropout(x1)
        x2 = x2 + self.dropout(x2)
        
        # Run the fully connected part of the layer
        y1 = x1 = self.norm1(x1)
        y1 = self.dropout(self.activation(self.linear1(y1)))
        y1 = self.dropout(self.linear2(y1))
        
        y2 = x2 = self.norm1(x2)
        y2 = self.dropout(self.activation(self.linear1(y2)))
        y2 = self.dropout(self.linear2(y2))

        return self.norm2(x1+y1), self.norm2(x2+y2)

class TransformerEncoderLayerCrossAttention(Module):
    """Self attention and feed forward network with skip connections.
    
    This transformer encoder layer implements the same encoder layer as
    PyTorch but is a bit more open for extension by receiving the attention
    implementation as a constructor argument.

    Arguments
    ---------
        attention: The attention implementation to use given as a nn.Module
        d_model: The input feature dimensionality
        d_ff: The dimensionality of the intermediate features after the
              attention (default: d_model*4)
        dropout: The dropout rate to apply to the intermediate features
                 (default: 0.1)
        activation: {'relu', 'gelu'} Which activation to use for the feed
                    forward part of the layer (default: relu)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1,
                 activation="relu", event_dispatcher=""):
        super(TransformerEncoderLayerCrossAttention, self).__init__()
        print('Using TransformerEncoderLayerCrossAttention')
        d_ff = d_ff or 4*d_model
        self.attention = attention
        self.linear1 = Linear(d_model, d_ff)
        self.linear2 = Linear(d_ff, d_model)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)
        
        self.hyper_layer = HypergraphLayer(d_model)
        
    def forward(self, x1, x2, attn_mask1=None, length_mask1=None, pos1=None, attn_mask2=None, length_mask2=None, pos2=None, hyper_adj2=None):
        """Apply the transformer encoder to the input x.

        Arguments
        ---------
            x: The input features of shape (N, L, E) where N is the batch size,
               L is the sequence length (padded) and E is d_model passed in the
               constructor.
            attn_mask: An implementation of fast_transformers.masking.BaseMask
                       that encodes where each element of x can attend to.
            length_mask: An implementation of
                         fast_transformers.masking.BaseMask that encodes how
                         many elements each sequence in the batch consists of.
        """
        # Normalize the masks
        N = x1.shape[0]
        L = x1.shape[1]
        attn_mask1 = attn_mask1 or FullMask(L, device=x1.device)
        length_mask1 = length_mask1 or \
            LengthMask(x1.new_full((N,), L, dtype=torch.int64))

        N = x2.shape[0]
        L = x2.shape[1]
        attn_mask2 = attn_mask2 or FullMask(L, device=x2.device)
        length_mask2 = length_mask2 or \
            LengthMask(x2.new_full((N,), L, dtype=torch.int64))
        
        x1, x2 = self.attention(
            x1, x1, x1,
            x2, x2, x2,
            attn_mask1=attn_mask1,
            query_lengths1=length_mask1,
            key_lengths1=length_mask1,
            pos1=pos1,
            attn_mask2=attn_mask2,
            query_lengths2=length_mask2,
            key_lengths2=length_mask2,
            pos2=pos2,
        )
        
        x1 = x1 + self.dropout(x1)
        x2 = x2 + self.dropout(x2)
        
        # Run the fully connected part of the layer
        y1 = x1 = self.norm1(x1)
        y1 = self.dropout(self.activation(self.linear1(y1)))
        y1 = self.dropout(self.linear2(y1))
        
        y2 = x2 = self.norm1(x2)
        y2 = self.dropout(self.activation(self.linear1(y2)))
        y2 = self.dropout(self.linear2(y2))

        return self.norm2(x1+y1), self.norm2(x2+y2)
        
class CrossAttentionEncoderBuilder(BaseTransformerEncoderBuilder):
    """Build a batch transformer encoder with Relative Rotary embeddings
    for training or processing of sequences all elements at a time.

    Example usage:

        builder = CrossAttentionEncoderBuilder()
        builder.n_layers = 12
        builder.n_heads = 8
        builder.feed_forward_dimensions = 1024
        builder.query_dimensions = 64
        builder.value_dimensions = 64
        builder.dropout = 0.1
        builder.attention_dropout = 0.1
        builder.attention_type = "linear"
        transformer = builder.get()
    """
    def _get_attention_builder(self):
        """Return an instance of the appropriate attention builder."""
        return AttentionBuilder()

    def _get_attention_layer_class(self):
        """Return the class for the layer that projects queries keys and
        values."""
        return CrossAttentionLayer

    def _get_encoder_class(self):
        """Return the class for the transformer encoder."""
        return TransformerEncoderCrossAttention

    def _get_encoder_layer_class(self):
        """Return the class for the transformer encoder layer."""
        return TransformerEncoderLayerCrossAttention
