import torch
import torch.nn as nn
import math
import dgl

from scipy import sparse as sp
import numpy as np
import networkx as nx
import torch.nn.functional as F

from model.GNN import GNN_node

class SmilesPositionalEncoding(nn.Module):
    def __init__(self, d_model, n_heads, max_len=5000):
        super(SmilesPositionalEncoding, self).__init__()
         
        self.name = 'SMILES'
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
 
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.n_heads = n_heads
        # pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, n_heads, emb]
        """
        b, s, _ = x.shape
        x = x.view(b, s, self.n_heads, -1)
        x = x.permute(0, 2, 1, 3)
        x = x + self.pe[:x.size(2), :] # broadcast. NOTE: should not be used when consider SMILES's personality
        print('padding_pos x', x.shape)
        print('padding_pos self.pe[:x.size(2), :]', self.pe[:x.size(2), :].shape)
        return x.permute(0, 2, 1, 3).view(b, s, -1)
        # x = x + self.pe[:x.size(1), :] # broadcast. NOTE: should not be used when consider SMILES's personality
        # return x

    
class GraphPositionalEncoding(nn.Module):
    def __init__(self, n_heads=None, d_model=None, max_num_spatial=100, pos_type=None, device=None):
        super(GraphPositionalEncoding, self).__init__()
        
        self.name = 'Graph'
        self.pos_type = pos_type
        if self.pos_type == 'adj_matrix':
            self.spatial_pos_encoder = nn.Embedding(max_num_spatial, n_heads, padding_idx=0)
            self.activation = F.gelu 
        elif self.pos_type == 'degree':
            self.spatial_pos_encoder = nn.Embedding(max_num_spatial, d_model, padding_idx=0)
            self.n_heads = n_heads
        elif self.pos_type in ['laplacian', 'hyper_laplacian']:
            self.pos_enc_dim = 6
            self.n_heads = n_heads
            self.spatial_pos_encoder = nn.Linear(self.pos_enc_dim, d_model)
            self.activation = F.gelu 
        elif self.pos_type == 'mpnn':
            # self.spatial_pos_encoder = GraphCNN(num_layers=5, num_mlp_layers=2, input_dim=n_heads*d_model, hidden_dim=n_heads*d_model, output_dim=n_heads*d_model, final_dropout=0.5, learn_eps=True, graph_pooling_type="average", neighbor_pooling_type="average", device=device)
            # self.spatial_pos_encoder = GAT_geom(num_hidden=n_heads*d_model)
            self.spatial_pos_encoder = GINNet_geom(num_hidden=n_heads*d_model)
        elif self.pos_type == 'GNN_edge':
            self.spatial_pos_encoder = GNN_node(num_layer=3, emb_dim=n_heads*d_model)
        else:
            raise ValueError('Error GraphPositionalEncoding.pos_type:{pos_type}'.format(pos_type=pos_type))
    
    def forward(self, x=None, QK=None, pos=None, twoDinfo=None, device=None):
        if self.pos_type == 'adj_matrix':
            return self.adj_matrix(QK=QK, pos=pos)
        elif self.pos_type == 'laplacian':
            return self.laplacian_positional_encoding(x=x, pos_tensor=pos, device=device)
        elif self.pos_type == 'degree':
            return self.degree_posiiton_encoding(x=x, adj=pos)
        elif self.pos_type == 'hyper_laplacian':
            return self.hypergraph_laplacian(x=x, pos_list=pos)
        elif self.pos_type == 'mpnn':
            x_list = []
            for i in range(len(pos)):
                adj_matrix = pos[i][1:-1, 1:-1]
                edge_index = torch.nonzero(adj_matrix, as_tuple=False).t()

                pre = x[i][0].unsqueeze(0)
                g_x = x[i][1:1+len(adj_matrix)]
                after = [x[i][j].unsqueeze(0) for j in range(1+len(adj_matrix), len(x[0]))]
                
                one_x = self.spatial_pos_encoder(g_x, edge_index)
                
                x_list.append(torch.cat([pre, g_x] + after, dim=0))
            return torch.stack(x_list, dim=0)
        elif self.pos_type == 'GNN_edge':
            b, l, d = x.shape
            x = x.view(-1, d)
            edge_index, edge_attr = twoDinfo
            # print('x', x.shape, x)
            # print('edge_index', edge_index.shape, edge_index)
            # print('edge_attr', edge_attr.shape, edge_attr)
            # print('edge_attr max ', max(edge_attr[:, 0]), max(edge_attr[:, 1]))
            # print('edge_index max', edge_index.shape, max(edge_index[0]), max(edge_index[1]))
            node_x = self.spatial_pos_encoder(x, edge_index, edge_attr)
            # print('node_x', node_x.shape)
            # print('node_x', node_x)
            node_x = node_x.view(b, l, d)
            # sys.exit()
            return node_x
    
    def degree_posiiton_encoding(self, x, adj):
        # print('degree_posiiton_encoding adj', adj)
        # degrees = torch.sum(adj, dim=1)
        # padding_pos = self.spatial_pos_encoder(degrees)
        # return x + padding_pos
        
        print('degree_posiiton_encoding x', x.shape)
    
        b, s, _ = x.shape
        x = x.view(b, s, self.n_heads, -1)
        x = x.permute(2, 0, 1, 3)
        degrees = torch.sum(adj, dim=1)
        # print('degree_posiiton_encoding degrees', degrees)
        padding_pos = self.spatial_pos_encoder(degrees)
        # print('degree_posiiton_encoding x', x.shape)
        # print('degree_posiiton_encoding padding_pos', padding_pos.shape)
        x = x + padding_pos # broadcast. NOTE: should not be used when consider SMILES's personality
        return x.permute(1, 2, 0, 3).view(b, s, -1)
            
    def adj_matrix(self, QK, pos):
        N, H, L, S = QK.shape
        # QK.shape = (N, H, L S), L=S
        # padding_pos = torch.zeros((N, L, S), dtype=torch.int32).to(QK.device)
        # for i, one_pos in enumerate(pos):
        #     padding_length = QK.shape[-1] - one_pos.shape[-1]
        #     padded_one_pos = torch.nn.functional.pad(one_pos, (0, padding_length, 0, padding_length))
        #     padding_pos[i, :, :] = padded_one_pos
        # # print('padding_pos:{padding_pos}'.format(padding_pos=type(padding_pos)))
        # print('pos:{p1}, pos.shape:{p2}, QK:{p3}'.format(p1=pos, p2=pos.shape, p3=QK.shape))
        # print('pos', pos)
        # tmp = self.spatial_pos_encoder(pos)
        # print('tmp:{p1}'.format(p1=tmp.shape))
        padding_pos = self.spatial_pos_encoder(pos).permute(0, 3, 1, 2) # (N, L, S) -> (N, L, S, H) -> (N, H, L, S)
        return QK + padding_pos

    def laplacian_positional_encoding(self, pos_tensor, x=None, device=None):
        """
            Graph positional encoding v/ Laplacian eigenvectors
            x: n_batch, length, n_head*emb
        """
        # print('pos_tensor', pos_tensor.shape, type(pos_tensor))
        # print('x', x.shape)
        batch_num = len(pos_tensor) # L = S
        length = max([pos.shape[0] for pos in pos_tensor])
        if device == None:
            device = x.device
        lap_matrix = torch.zeros((batch_num, length, self.pos_enc_dim)).to(device)
        for i in range(batch_num):
            # print(i, pos)
            pos = pos_tensor[i]
            src, des = pos.to_sparse().indices()
            g = dgl.graph((src, des))
            # Laplacian
            A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
            N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
            L = sp.eye(g.number_of_nodes()) - N * A * N
            L = L.toarray()
            
            # Eigenvectors with scipy
            # print(L-L.T)

            EigVal, EigVec = sp.linalg.eigsh(L, k=self.pos_enc_dim+1, which='LM', ncv=20, tol=1e-5) # for 40 PEs
            # EigVal, EigVec = sp.linalg.eigsh(L, k=self.pos_enc_dim+1, which='SA', tol=1e-2) # for 40 PEs
            EigVec = EigVec[:, EigVal.argsort()] # increasing order
            EigVec = EigVec[:,1:self.pos_enc_dim+1]
            lap_matrix[i, :len(EigVec), :len(EigVec[0])] = torch.from_numpy(np.real(EigVec).astype(np.float32))
        lap_matrix = self.activation(self.spatial_pos_encoder(lap_matrix))
        lap_matrix = lap_matrix.unsqueeze(2)
        if x == None:
            return lap_matrix
        else:
            x = x.view(batch_num, length, self.n_heads, -1)
            # print('x', x.shape)
            # print('lap_matrix', lap_matrix.shape)
            output = x + lap_matrix
            output = output.view(batch_num, length, -1)
            # print('output', output.shape)
            return output
        
    
    def laplacian_positional_encoding_list(self, pos_list, x=None, device=None):
        """
            Graph positional encoding v/ Laplacian eigenvectors
            x: n_batch, length, n_head*emb
        """
        print('pos_list', pos_list.shape, type(pos_list))
        print('x', x.shape)
        batch_num = len(pos_list) # L = S
        length = max([pos.shape[0] for pos in pos_list])
        if device == None:
            device = x.device
        lap_matrix = torch.zeros((batch_num, length, self.pos_enc_dim)).to(device)
        for i, pos in enumerate(pos_list):
            # print(i, pos)
            src, des = pos.to_sparse().indices()
            g = dgl.graph((src, des))
            # Laplacian
            A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
            N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
            L = sp.eye(g.number_of_nodes()) - N * A * N
            L = L.toarray()
            
            # Eigenvectors with scipy
            # print(L-L.T)
            EigVal, EigVec = sp.linalg.eigsh(L, k=self.pos_enc_dim+1, which='SA', tol=1e-2) # for 40 PEs
            EigVec = EigVec[:, EigVal.argsort()] # increasing order
            EigVec = EigVec[:,1:self.pos_enc_dim+1]
            lap_matrix[i, :len(EigVec), :len(EigVec[0])] = torch.from_numpy(np.real(EigVec).astype(np.float32))
        lap_matrix = self.activation(self.spatial_pos_encoder(lap_matrix))
        lap_matrix = lap_matrix.unsqueeze(2)
        if x == None:
            return lap_matrix
        else:
            x = x.view(batch_num, length, self.n_heads, -1)
            output = x + lap_matrix
            return output.view(batch_num, length, -1)

    def hypergraph_laplacian(self, x, pos_list):
        """
        Compute the Laplacian matrix of a hypergraph.
        H is the incidence matrix where H[i, j] = 1 if vertex i is in hyperedge j.
        """
        # print('pos_list',pos_list)
        batch_num = len(pos_list) # L = S
        length = max([pos.shape[0] for pos in pos_list])
        lap_matrix = torch.zeros((batch_num, length, self.pos_enc_dim)).to(x.device)
        for i, H in enumerate(pos_list):
            H = H.float()
            D = torch.diag(torch.sum(H, dim=1))
            W = torch.diag(torch.sum(H, dim=0))
            W_inv = torch.inverse(W)
            L = D - torch.mm(torch.mm(H, W_inv), torch.transpose(H, 0, 1))

            eigvals, eigvecs = torch.linalg.eig(L)
            sorted_indices = torch.argsort(eigvals.real)  # Use the real part for sorting
            eigvals = eigvals[sorted_indices]
            eigvecs = eigvecs[:, sorted_indices]

            lap_matrix[i, :len(eigvecs), :len(eigvecs[0])] = eigvecs[:, :self.pos_enc_dim].real
        lap_matrix = self.activation(self.spatial_pos_encoder(lap_matrix))
        return x + lap_matrix
    

    
class RotaryEmbedding(torch.nn.Module):
    
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.seq_len_cached = 0 
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            #if seq_len > self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[None,:, None, :]
            self.sin_cached = emb.sin()[None,:, None, :]
            #else:
            #    cos_return = self.cos_cached[..., :seq_len]
            #    sin_return = self.sin_cached[..., :seq_len]
            #    return cos_return, sin_return
                
        return self.cos_cached, self.sin_cached


# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=x1.ndim - 1) # dim=-1 triggers a bug in earlier torch versions

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)




import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = node_features
        self.edge_mat = 0

        self.max_neighbor = 0
        
###MLP with lienar output
class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''
    
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            #Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()
        
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)
        
class GraphCNN(nn.Module):
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim, output_dim, final_dropout, learn_eps, graph_pooling_type, neighbor_pooling_type, device):
        '''
            num_layers: number of layers in the neural networks (INCLUDING the input layer)
            num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            final_dropout: dropout ratio on the final linear layer
            learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether. 
            neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
            graph_pooling_type: how to aggregate entire nodes in a graph (mean, average)
            device: which device to use
        '''

        super(GraphCNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.graph_pooling_type = graph_pooling_type
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.eps = nn.Parameter(torch.zeros(self.num_layers-1))

        ###List of MLPs
        self.mlps = torch.nn.ModuleList()

        ###List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # #Linear function that maps the hidden representation at dofferemt layers into a prediction score
        # self.linears_prediction = torch.nn.ModuleList()
        # for layer in range(num_layers):
        #     if layer == 0:
        #         self.linears_prediction.append(nn.Linear(input_dim, output_dim))
        #     else:
        #         self.linears_prediction.append(nn.Linear(hidden_dim, output_dim))


    def __preprocess_neighbors_maxpool(self, batch_graph):
        ###create padded_neighbor_list in concatenated graph

        #compute the maximum number of neighbors within the graphs in the current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]


        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                #add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                #padding, dummy data is assumed to be stored in -1
                pad.extend([-1]*(max_deg - len(pad)))

                #Add center nodes in the maxpooling if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)


    def __preprocess_neighbors_sumavepool(self, batch_graph):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        #Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.

        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))

        return Adj_block.to(self.device)


    def __preprocess_graphpool(self, batch_graph):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        
        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ###average pooling
            if self.graph_pooling_type == "average":
                elem.extend([1./len(graph.g)]*len(graph.g))
            
            else:
            ###sum pooling
                elem.extend([1]*len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        
        return graph_pool.to(self.device)

    def maxpool(self, h, padded_neighbor_list):
        ###Element-wise minimum will never affect max-pooling

        dummy = torch.min(h, dim = 0)[0]
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        pooled_rep = torch.max(h_with_dummy[padded_neighbor_list], dim = 1)[0]
        return pooled_rep


    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes separately by epsilon reweighting. 

        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            # print('Adj_block', Adj_block)
            # print('h', h)
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h


    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        ###pooling neighboring nodes and center nodes altogether  
            
        if self.neighbor_pooling_type == "max":
            ##If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            #If sum or average pooling
            pooled = torch.spmm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                #If average pooling
                degree = torch.spmm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        #representation of neighboring and center nodes 
        pooled_rep = self.mlps[layer](pooled)

        h = self.batch_norms[layer](pooled_rep)

        #non-linearity
        h = F.relu(h)
        return h

    def forward(self, x, adj):
        B, S, D = x.shape
        print('adj', adj)

        batch_graph = []
        node_num_list = []
        for i in range(len(adj)):
            adj_matrix = adj[i].cpu().numpy()[1:-1, 1:-1]
            
            node_num_list.append(len(adj_matrix))
            g_x = x[i][1:1+len(adj_matrix)]
            print('g_x', g_x.shape)
            print('adj_matrix', len(adj_matrix), len(adj_matrix[0]))
            G = nx.Graph()

            G.add_nodes_from(range(len(adj_matrix)))

            edges = np.argwhere(adj_matrix > 0)
            for edge in edges:
                G.add_edge(edge[0], edge[1], weight=adj_matrix[edge[0], edge[1]])
            g = S2VGraph(G, 0, node_features=g_x)
            #add labels and edge_mat       
            g.neighbors = [[] for i in range(len(g.g))]
            for i, j in g.g.edges():
                g.neighbors[i].append(j)
                g.neighbors[j].append(i)
            degree_list = []
            for i in range(len(g.g)):
                g.neighbors[i] = g.neighbors[i]
                degree_list.append(len(g.neighbors[i]))
            g.max_neighbor = max(degree_list)

            # g.label = label_dict[g.label]

            edges = [list(pair) for pair in g.g.edges()]
            edges.extend([[i, j] for j, i in edges])

            deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
            g.edge_mat = torch.LongTensor(edges).transpose(0,1)
            # print(g.g)

            batch_graph.append(g)
        # print('batch_graph', len(batch_graph))                 
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).to(self.device)
        graph_pool = self.__preprocess_graphpool(batch_graph)

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = self.__preprocess_neighbors_maxpool(batch_graph)
        else:
            Adj_block = self.__preprocess_neighbors_sumavepool(batch_graph)

        #list of hidden representation at each layer (including input)
        hidden_rep = [X_concat]
        h = X_concat

        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block = Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list = padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block = Adj_block)

            hidden_rep.append(h)

        # score_over_layer = 0
        # #perform pooling over all nodes in each graph in every layer
        # for layer, h in enumerate(hidden_rep):
        #     # pooled_h = torch.spmm(graph_pool, h)
        #     pooled_h = h
        #     score_over_layer += F.dropout(self.linears_prediction[layer](pooled_h), self.final_dropout, training = self.training)
        # score_over_layer = score_over_layer.view(B, S, D )
        # score_over_layer = torch.where(torch.isnan(score_over_layer), x, score_over_layer)
        
        print('h', h)
        h = h.view(B, S, D)
        print('h', h)
        # h = torch.where(torch.isnan(h), x, h)
        output = 1
        
        # print('score_over_layer', score_over_layer)
#         sys.exit()
        
        return h
    
    
from torch_geometric.nn import GCNConv, GATConv, GINConv
#python GAT_RARE.py --dataset wisconsin --dataset_split "splits/wisconsin_split_0.6_0.2_0.npz" --run_id 0 --seed 10 --RL_policy ppo --RL_GNN_steps 5 --RL_add_max_neighbor_num 2 --RL_del_max_neighbor_num 1 --total_timesteps 500 --dropout_rate 0.5 --num_hidden 48 --learning_rate 0.05 --num_heads_layer_one 8 --num_heads_layer_two 1 --weight_decay_layer_one 5e-06 --weight_decay_layer_two 5e-06

class GAT_geom(nn.Module):
    def __init__(self, num_hidden, num_heads_layer_one=8, num_heads_layer_two=1, dropout_rate=0.5):
        super(GAT_geom, self).__init__()
        self.conv1 = GATConv(
            num_hidden,
            num_hidden,
            heads=num_heads_layer_one,
            dropout=dropout_rate)
        self.conv2 = GATConv(
            num_hidden * num_heads_layer_one,
            num_hidden,
            heads=num_heads_layer_two,
            concat=False,
            dropout=dropout_rate)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
    def forward(self, features, edge_index):
        x = F.relu(self.conv1(features, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
class GINNet_geom(nn.Module):
    def __init__(self, num_hidden):
        super(GINNet_geom, self).__init__()
        nn1 = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden))
        self.gin1 = GINConv(nn1, train_eps=True)
        nn2 = nn.Sequential(nn.Linear(num_hidden, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_hidden))
        self.gin2 = GINConv(nn2, train_eps=True)
        
    def forward(self, features, edge_index):
        x = F.relu(self.gin1(features, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.gin2(x, edge_index)
        return x   