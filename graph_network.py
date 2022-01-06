import torch
import torch.nn as nn
from torch_scatter import scatter
from dataloader import NCDataset
from models import GCN, GAT
from interation_net import EdgeModel, NodeModel, MetaLayer
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        if num_hidden_layers == 0:
            self.lins.append(nn.Linear(input_size, output_size))
        else:
            self.lins.append(nn.Linear(input_size, hidden_size))
            for _ in range(num_hidden_layers - 1):
                self.lins.append(nn.Linear(hidden_size, hidden_size))
            self.lins.append(nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for lin in self.lins:
            x = lin(x)
        return x

class EncodeProcessDecode(nn.Module):
    def __init__(self,
                 node_input_size,
                 edge_input_size,
                 latent_size,
                 mlp_hidden_size,
                 mlp_num_hidden_layers,
                 num_message_passing_steps,
                 output_size,
                 device,
                 args,
                 name="EncodeProcessDecode"):
        super().__init__()

        self._node_input_size = node_input_size
        self._edge_input_size = edge_input_size
        self._latent_size = latent_size
        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers
        self._num_message_passing_steps = num_message_passing_steps
        self._output_size = output_size
        self.device = device
        self.args = args

        self._network_builder()

    def forward(self, input_graph):
        # Encode the input graph.
        latent_graph_0 = self._encode(input_graph)
        # Do `m` message passing steps in the latent graphs.
        latent_graph_m = self._process(latent_graph_0)
        # Decode from the last latent graph.
        return self._decode(latent_graph_m)

    def _network_builder(self):
        """Builds the networks."""
        def build_mlp_with_layer_norm(input_size):
            mlp = MLP(
                input_size=input_size,
                hidden_size=self._mlp_hidden_size,
                num_hidden_layers=self._mlp_num_hidden_layers,
                output_size=self._latent_size
            )
            return nn.Sequential(*[mlp, nn.LayerNorm(self._latent_size)])

        self.node_encoder = build_mlp_with_layer_norm(self._node_input_size)
        self.edge_encoder = build_mlp_with_layer_norm(self._edge_input_size)

        self._processor_networks = nn.ModuleList()
        for _ in range(self._num_message_passing_steps):
            
            if self.args.gnn_type == 'gcn':
                self._processor_networks.append(GCN(self._latent_size, self.args.hidden_channels,
                                                    self._latent_size, self.args.num_gnn_layers,
                                                    self.args.dropout, use_bn=self.args.use_bn).to(self.device))
            elif self.args.gnn_type == 'gat':
                self._processor_networks.append(GAT(self._latent_size, self.args.hidden_channels,
                                                    self._latent_size, self.args.num_gnn_layers,
                                                    self.args.dropout, self.args.use_bn,
                                                    self.args.gat_heads, self.args.out_heads).to(self.device))
            elif self.args.gnn_type == 'interaction_net':
                self._processor_networks.append(
                                        MetaLayer(
                                            edge_model=EdgeModel(build_mlp_with_layer_norm(input_size=self._latent_size*3)),
                                            node_model=NodeModel(build_mlp_with_layer_norm(input_size=self._latent_size*2)),
                                            global_model=None).to(self.device))
            else:
                raise ValueError("Not recognized GNN model!")

        self._decoder_network = MLP(
            input_size=self._latent_size,
            hidden_size=self._mlp_hidden_size,
            num_hidden_layers=self._mlp_num_hidden_layers,
            output_size=self._output_size)

    def _encode(self, input_graph):
        # TODO adapt encoder's input size
        if input_graph.graph['global'] is not None:
            input_graph.graph['node_feat'] = torch.cat([input_graph.graph['node_feat'], input_graph.graph['global']], dim=-1)

        latent_graph_0 = NCDataset("latent_graph_0")
        latent_graph_0.graph = {
            'node_feat': self.node_encoder(input_graph.graph['node_feat']),
            'edge_feat': self.edge_encoder(input_graph.graph['edge_feat']),
            'global': None,
            'n_node': input_graph.graph['n_node'],
            'n_edge': input_graph.graph['n_edge'],
            'edge_index': input_graph.graph['edge_index'].to(input_graph.graph['n_node'].device),
        }
        num_nodes = torch.sum(latent_graph_0.graph['n_node']).item()

        latent_graph_0.graph['node_feat'] += scatter(latent_graph_0.graph['edge_feat'], latent_graph_0.graph['edge_index'][0],
                                                    dim=0, dim_size=num_nodes, reduce='mean')
        return latent_graph_0

    def _process(self, latent_graph_0):
        latent_graph_prev_k = latent_graph_0
        latent_graph_k = latent_graph_0
        for processor_network_k in self._processor_networks:
            latent_graph_k = self._process_step(
                processor_network_k, latent_graph_prev_k)
            latent_graph_prev_k = latent_graph_k

        latent_graph_m = latent_graph_k
        return latent_graph_m

    def _process_step(self, processor_network_k, latent_graph_prev_k):
        if self.args.gnn_type == 'interaction_net':
            new_node_feature, new_edge_feature = processor_network_k(latent_graph_prev_k.graph)
            latent_graph_k = NCDataset('latent_graph_k')
            latent_graph_k.graph = {
                'node_feat': latent_graph_prev_k.graph['node_feat'] + new_node_feature,
                'edge_feat': latent_graph_prev_k.graph['edge_feat'] + new_edge_feature,
                'global': None,
                'n_node': latent_graph_prev_k.graph['n_node'],
                'n_edge': latent_graph_prev_k.graph['n_edge'],
                'edge_index': latent_graph_prev_k.graph['edge_index']
            }
        else:
            new_node_feature = processor_network_k(latent_graph_prev_k)
            latent_graph_k = NCDataset('latent_graph_k')
            latent_graph_k.graph = {
                'node_feat': latent_graph_prev_k.graph['node_feat'] + new_node_feature,
                'edge_feat': None,
                'global': None,
                'n_node': latent_graph_prev_k.graph['n_node'],
                'n_edge': latent_graph_prev_k.graph['n_edge'],
                'edge_index': latent_graph_prev_k.graph['edge_index']
            }

        return latent_graph_k

    def _decode(self, latent_graph):
        return self._decoder_network(latent_graph.graph['node_feat'])


