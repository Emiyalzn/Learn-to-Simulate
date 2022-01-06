import torch
from torch_scatter import scatter
from torch_geometric.nn import MetaLayer

class EdgeModel(torch.nn.Module):
    def __init__(self,edge_mlp):
        super().__init__()
        self.edge_mlp = edge_mlp

    def forward(self, x, edge_index, edge_attr):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        edge_index = edge_index.permute(1,0)
        sender_node_feature = x[edge_index[0]]
        receive_node_feature = x[edge_index[1]]
        out = torch.cat([sender_node_feature, receive_node_feature, edge_attr] , dim = - 1)
        
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self,node_mlp):
        super().__init__()
        self.node_mlp = node_mlp

    def forward(self, x, edge_index, edge_attr):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        num_nodes = x.size(0)
        nodes_to_collect = []
        nodes_to_collect.append(x)
        
        receive_edge_attr = scatter(edge_attr, edge_index[1], dim=0, dim_size=num_nodes, reduce='mean')
        nodes_to_collect.append(receive_edge_attr)
    
        nodes_to_collect = torch.cat(nodes_to_collect,dim = -1)
        
        return self.node_mlp(nodes_to_collect)


class GlobalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index, edge_attr, u, batch):
        '''
            Global Model not support now
        '''
        pass


from typing import Optional, Tuple

class MetaLayer(torch.nn.Module):
    def __init__(self, edge_model=None, node_model=None, global_model=None):
        super().__init__()
        self.edge_model = edge_model
        self.node_model = node_model
        self.global_model = global_model

        self.reset_parameters() 

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()


    def forward(self, graph):
        x = graph['node_feat']
        edge_index = graph['edge_index']
        edge_attr = graph['edge_feat']
        
        if self.edge_model is not None:
            edge_attr = self.edge_model(x, edge_index, edge_attr)

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr)

        assert self.global_model is None, 'Global model is not support!'

        return x, edge_attr
    