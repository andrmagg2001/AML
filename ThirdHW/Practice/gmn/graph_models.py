import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MetaLayer
from torch_geometric.nn.pool import max_pool_x, avg_pool_x, global_max_pool, global_mean_pool
from torch_scatter import scatter

class EdgeModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True):
        super().__init__()
        if activation:
            self.edge_mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            self.edge_mlp = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, src, dest, edge_attr, u, batch):
        # **IMPORTANT: YOU ARE NOT ALLOWED TO USE FOR LOOPS!**
        # src, dest: [E, F_x], where E is the number of edges. src is the source node features and dest is the destination node features of each edge.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: only here it will have shape [E] with max entry B - 1, because here it indicates the graph index for each edge.

        '''
        Add your code below
        '''
        u_batch = u[batch]
        edge_input = torch.cat([src, dest, edge_attr, u_batch], dim=1)

        # print("EDGE MODEL")
        # print(f"src: {src.shape}")
        # print(f"dest: {dest.shape}")
        # print(f"edge_attr: {edge_attr.shape}")
        # print(f"u_batch: {u_batch.shape}")
        # print(f"edge_input: {edge_input.shape}")
        # print("#########################################")

        updated_edge = self.edge_mlp(edge_input)

        return updated_edge

class NodeModel(nn.Module):
    def __init__(self, in_dim_mlp1, in_dim_mlp2, out_dim, activation=True, reduce='sum'):
        super().__init__()
        self.reduce = reduce
        if activation:
            self.node_mlp_1 = nn.Sequential(nn.Linear(in_dim_mlp1, out_dim), nn.ReLU())
            self.node_mlp_2 = nn.Sequential(nn.Linear(in_dim_mlp2, out_dim), nn.ReLU())
        else:
            self.node_mlp_1 = nn.Sequential(nn.Linear(in_dim_mlp1, out_dim))
            self.node_mlp_2 = nn.Sequential(nn.Linear(in_dim_mlp2, out_dim))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # **IMPORTANT: YOU ARE NOT ALLOWED TO USE FOR LOOPS!**
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        '''
        Add your code below
        '''
        u_batch = u[batch]
        ### MLP1 section ###
        dest, src = edge_index
        edge_info = torch.cat([x[src], edge_attr], dim=1)
    
        # print("NODE MODEL")
        # print("MLP1")
        # print(f"x[src]: {x[src].shape}")
        # print(f"edge_attr: {edge_attr.shape}")
        # print(f"INPUT edge_info: {edge_info.shape}")

        edge_messages = self.node_mlp_1(edge_info)


        # aggregation of all messages
        aggregated_messages = scatter(edge_messages, dest, dim=0, reduce=self.reduce)

        ### MLP2 section ###
        node_info = torch.cat([x, aggregated_messages, u_batch], dim=1)
        # print("MLP2")
        # print(f"x: {x.shape}")
        # print(f"aggregated_messages: {aggregated_messages.shape}")
        # print(f"u_batch: {u_batch.shape}")
        # print(f"INPUT node_info: {node_info.shape}")
        # print("#########################################")

        updated_node_features = self.node_mlp_2(node_info)


        return updated_node_features


class GlobalModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True, reduce='sum'):
        super().__init__()
        if activation:
            self.global_mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            self.global_mlp = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.reduce = reduce

    def forward(self, x, edge_index, edge_attr, u, batch):
        #**IMPORTANT: YOU ARE NOT ALLOWED TO USE FOR LOOPS!**
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        '''
        Add your code below
        '''
        
        # Aggregate node features
        # u_batch = u[batch]
        node_aggregated = scatter(x, batch, dim=0, reduce=self.reduce)
        
        # Aggregate edges features
        edge_batch = batch[edge_index[0]]
        edge_aggregated = scatter(edge_attr, edge_batch, dim=0, reduce=self.reduce)
        
        # Concat the info
        global_info = torch.cat([node_aggregated, edge_aggregated, u], dim=1)
        # print("GLOBAL MODEL")
        # print(f"node_aggregated: {node_aggregated.shape}")
        # print(f"edge_aggregated: {edge_aggregated.shape}")
        # print(f"u: {u.shape}")
        # print(f"global_info: {global_info.shape}")
        # print("###########################################################")
        
        # Pass to MLP
        updated_global = self.global_mlp(global_info)
        
        return updated_global

class MPNN(nn.Module):

    def __init__(self, node_in_dim, edge_in_dim, global_in_dim, hidden_dim, node_out_dim, edge_out_dim, global_out_dim, num_layers,
                use_bn=True, dropout=0.0, reduce='sum'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        self.global_norms = nn.ModuleList()
        self.use_bn = use_bn
        self.dropout = dropout
        self.reduce = reduce

        assert num_layers >= 2

        '''
        Instantiate the first layer models with correct parameters below
        '''

        edge_model = EdgeModel(in_dim=edge_in_dim + 2 * node_in_dim + global_in_dim, out_dim=hidden_dim)
        node_model = NodeModel(in_dim_mlp1=node_in_dim + hidden_dim, in_dim_mlp2=node_in_dim + hidden_dim + global_in_dim, out_dim=hidden_dim)
        global_model = GlobalModel(in_dim=hidden_dim + hidden_dim + global_in_dim, out_dim=hidden_dim)
        self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))
        self.node_norms.append(nn.BatchNorm1d(hidden_dim))
        self.edge_norms.append(nn.BatchNorm1d(hidden_dim))
        self.global_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers-2):
            '''
            Add your code below
            '''
            # add batch norm after each MetaLayer
            # edge_model = EdgeModel(in_dim=edge_in_dim + 2 * node_in_dim + global_in_dim, out_dim=hidden_dim)
            # node_model = NodeModel(in_dim_mlp1=node_in_dim + hidden_dim, in_dim_mlp2=node_in_dim + hidden_dim + global_in_dim, out_dim=hidden_dim)
            # global_model = GlobalModel(in_dim=hidden_dim + hidden_dim + global_in_dim, out_dim=hidden_dim)
            # self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))
            # self.node_norms.append(nn.BatchNorm1d(hidden_dim))
            # self.edge_norms.append(nn.BatchNorm1d(hidden_dim))
            # self.global_norms.append(nn.BatchNorm1d(hidden_dim))
            edge_model = EdgeModel(in_dim= 4 * hidden_dim, out_dim=hidden_dim)
            node_model = NodeModel(in_dim_mlp1= 2 * hidden_dim, in_dim_mlp2= 3 * hidden_dim, out_dim=hidden_dim)
            global_model = GlobalModel(in_dim= 3 * hidden_dim, out_dim=hidden_dim)
            self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))
            self.node_norms.append(nn.BatchNorm1d(hidden_dim))
            self.edge_norms.append(nn.BatchNorm1d(hidden_dim))
            self.global_norms.append(nn.BatchNorm1d(hidden_dim))


        '''
        Add your code below
        '''
        # last MetaLayer without batch norm and without using activation functions
        # edge_model = EdgeModel(in_dim=edge_in_dim + 2 * node_in_dim + global_in_dim, out_dim=edge_out_dim, activation=False)
        # node_model = NodeModel(in_dim_mlp1=node_in_dim + hidden_dim, in_dim_mlp2=node_in_dim + hidden_dim + global_in_dim, out_dim=node_out_dim, activation=False)
        # global_model = GlobalModel(in_dim=hidden_dim + hidden_dim + global_in_dim, out_dim=global_out_dim, activation=False)
        # self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))
        edge_model = EdgeModel(in_dim= 4 * hidden_dim, out_dim=edge_out_dim, activation=False)
        node_model = NodeModel(in_dim_mlp1= hidden_dim + node_out_dim, in_dim_mlp2= 2 * hidden_dim + node_out_dim, out_dim=node_out_dim, activation=False)
        global_model = GlobalModel(in_dim= 2 * hidden_dim, out_dim=global_out_dim, activation=False)
        self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))


    def forward(self, x, edge_index, edge_attr, u, batch, *args):

        for i, conv in enumerate(self.convs):
            '''
            Add your code below
            '''
            # print()
            # print(f"MODEL INPUT SHAPES:")
            # print(f"x: {x.shape}")
            # print(f"edge_index: {edge_index.shape}")
            # print(f"edge_attr: {edge_attr.shape}")
            # print(f"u: {u.shape}")
            # print(f"batch: {batch.shape}")
            # print("########################################################")
            # Apply layer
            x, edge_attr, u = conv(x, edge_index, edge_attr, u, batch)

            # print(f"After MetaLayer {i}:")
            # print(f"x: {x.shape}, edge_attr: {edge_attr.shape}, u: {u.shape}")

            if i != len(self.convs)-1 and self.use_bn:
                '''
                Add your code below this line, but before the dropout
                '''
                # Apply batch_norm
                x = self.node_norms[i](x)
                edge_attr = self.edge_norms[i](edge_attr)
                u = self.global_norms[i](u)

            # Apply dropout
            x = F.dropout(x, p=self.dropout, training=self.training)
            edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)
            u = F.dropout(u, p=self.dropout, training=self.training)

        return x, edge_attr, u