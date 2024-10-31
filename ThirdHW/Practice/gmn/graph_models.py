# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MetaLayer
from torch_geometric.nn.pool import max_pool_x, avg_pool_x, global_max_pool, global_mean_pool
from torch_scatter import scatter

class EdgeModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True):
        super().__init__()
        # replace this with the class EdgeModel implemented by you in the theory part
        if activation:
            self.edge_mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            self.edge_mlp = nn.Sequential(nn.Linear(in_dim, out_dim))
        

    def forward(self, src, dest, edge_attr, u, batch):
        # replace this with the forward function of the EdgeModel class implemented in the theory part
        u_batch = u[batch]  # Broadcast u based on the batch index of each edge
        edge_input = torch.cat([src, dest, edge_attr, u_batch], dim=1)  # Concatenate features
        updated_edge = self.edge_mlp(edge_input)  # Apply MLP to compute updated edge features

        return updated_edge

class NodeModel(nn.Module):
    def __init__(self, in_dim_mlp1, in_dim_mlp2, out_dim, activation=True, reduce='sum'):
        super().__init__()
        # replace this with the class NodeModel implemented by you in the theory part
        self.reduce = reduce
        if activation:
            self.node_mlp_1 = nn.Sequential(nn.Linear(in_dim_mlp1, out_dim), nn.ReLU())
            self.node_mlp_2 = nn.Sequential(nn.Linear(in_dim_mlp2, out_dim), nn.ReLU())
        else:
            self.node_mlp_1 = nn.Sequential(nn.Linear(in_dim_mlp1, out_dim))
            self.node_mlp_2 = nn.Sequential(nn.Linear(in_dim_mlp2, out_dim))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # replace this with the forward function of the NodeModel class implemented in the theory part
        u_batch = u[batch]
        ### MLP1 section ###
        dest, src = edge_index
        edge_info = torch.cat([x[src], edge_attr], dim=1)
        

        edge_messages = self.node_mlp_1(edge_info)


        # aggregation of all messages
        aggregated_messages = scatter(edge_messages, dest, dim=0, reduce=self.reduce)

        ### MLP2 section ###
        node_info = torch.cat([x, aggregated_messages, u_batch], dim=1)

        updated_node_features = self.node_mlp_2(node_info)

        return updated_node_features


class GlobalModel(nn.Module):
    def __init__(self, in_dim, out_dim, reduce='sum', activation=True):
        super().__init__()
        # replace this with the class GlobalModel implemented by you in the theory part
        if activation:
            self.global_mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            self.global_mlp = nn.Sequential(nn.Linear(in_dim, out_dim))

        self.reduce = reduce
    def forward(self, x, edge_index, edge_attr, u, batch):
        # replace this with the forward function of the GlobalModel class implemented in the theory part
        u_batch = u[batch]
        node_aggregated = scatter(x, batch, dim=0, reduce=self.reduce)
        
        # Aggregate edges features
        edge_batch = batch[edge_index[0]]
        edge_aggregated = scatter(edge_attr, edge_batch, dim=0, reduce=self.reduce)
        
        # Concatenare le informazioni dei nodi aggregati, degli archi aggregati e del vettore globale u
        global_info = torch.cat([node_aggregated, edge_aggregated, u_batch], dim=1)
        
        # Pass to MLP
        updated_global = self.global_mlp(global_info)
        
        return updated_global

class MPNN(nn.Module):

    def __init__(self, node_in_dim, edge_in_dim, global_in_dim, hidden_dim, node_out_dim, edge_out_dim, global_out_dim, num_layers,
                use_bn=True, dropout=0.0, reduce='sum'):
        super().__init__()
        # replace this with the class MPNN implemented by you in the theory part
        self.convs = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        self.global_norms = nn.ModuleList()
        self.use_bn = use_bn
        self.dropout = dropout
        self.reduce = reduce

        assert num_layers >= 2

        edge_model = EdgeModel(in_dim=edge_in_dim + 2 * node_in_dim + global_in_dim, out_dim=hidden_dim)
        node_model = NodeModel(in_dim_mlp1=node_in_dim + hidden_dim, in_dim_mlp2=node_in_dim + hidden_dim + global_in_dim, out_dim=hidden_dim)
        global_model = GlobalModel(in_dim=node_in_dim + edge_in_dim + global_in_dim, out_dim=hidden_dim)
        self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))
        self.node_norms.append(nn.BatchNorm1d(hidden_dim))
        self.edge_norms.append(nn.BatchNorm1d(hidden_dim))
        self.global_norms.append(nn.BatchNorm1d(hidden_dim))



        for _ in range(num_layers-2):
            edge_model = EdgeModel(in_dim=edge_in_dim + 2 * node_in_dim + global_in_dim, out_dim=hidden_dim)
            node_model = NodeModel(in_dim_mlp1=node_in_dim + hidden_dim, in_dim_mlp2=node_in_dim + hidden_dim + global_in_dim, out_dim=hidden_dim)
            global_model = GlobalModel(in_dim=node_in_dim + edge_in_dim + global_in_dim, out_dim=hidden_dim)
            self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))
            self.node_norms.append(nn.BatchNorm1d(hidden_dim))
            self.edge_norms.append(nn.BatchNorm1d(hidden_dim))
            self.global_norms.append(nn.BatchNorm1d(hidden_dim))


        edge_model = EdgeModel(in_dim=edge_in_dim + 2 * node_in_dim + global_in_dim, out_dim=hidden_dim, activation=False)
        node_model = NodeModel(in_dim_mlp1=node_in_dim + hidden_dim, in_dim_mlp2=node_in_dim + hidden_dim + global_in_dim, out_dim=hidden_dim, activation=False)
        global_model = GlobalModel(in_dim=node_in_dim + edge_in_dim + global_in_dim, out_dim=hidden_dim, activation=False)
        self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))

        
        

    def forward(self, x, edge_index, edge_attr, u, batch, *args):
        # replace this with the forward function of the MPNN class implemented in the theory part
        for i, conv in enumerate(self.convs):
            x, edge_attr, u = conv(x, edge_index, edge_attr, u, batch)

            if i != len(self.convs)-1 and self.use_bn:
                x = self.node_norms[i](x)
                edge_attr = self.edge_norms[i](edge_attr)
                u = self.global_norms[i](u)



            x = F.dropout(x, p=self.dropout, training=self.training)
            edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)
            u = F.dropout(u, p=self.dropout, training=self.training)

        return x, edge_attr, u