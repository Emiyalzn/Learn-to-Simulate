import numpy as np
from sklearn import neighbors
import torch

def _compute_connectivity(positions, radius, add_self_edges=True):
    """Get the indices of connected edges with radius connectivity.
    Args:
        positions: Positions of nodes in the graph. [num_nodes_in_graph, num_dims]
        radius: Radius of connectivity.
        add_self_edges: Whether to include self edges or not.
    Returns:
        senders indices [num_edges_in_graph]
        receivers indices [num_edges_in_graph]
    """
    tree = neighbors.KDTree(positions)
    receivers_list = tree.query_radius(positions, r=radius)
    num_nodes = len(positions)
    senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)

    if not add_self_edges:
        mask = senders != receivers
        senders = senders[mask]
        receivers = receivers[mask]

    return senders, receivers

def compute_connectivity_for_batch(positions, n_node, radius, add_self_edges=True, device='cpu'):
    """`compute_connectivity` for a batch of graphs.
    Args:
        positions: [num_nodes_in_batch, num_dims]
        n_node: number of nodes for each graph in the batch. [num_graphs_in_batch]
        radius: Radius of connectivity.
        add_self_edges: Whether to include self edges or not.
    Returns:
        sender indices [num_edges_in_batch]
        receiver indices [num_edges_in_batch]
        number of edges per graph [num_graphs_in_batch]
    """
    positions_per_graph_list = np.split(positions, np.cumsum(n_node[:-1]), axis=0)
    receivers_list = []
    senders_list = []
    n_edge_list = []
    num_nodes_in_previous_graphs = 0

    # Compute connectivity for each graph in the batch.
    for positions_graph_i in positions_per_graph_list:
        senders_graph_i, receivers_graph_i = _compute_connectivity(
            positions_graph_i, radius, add_self_edges)
        num_edges_graph_i = len(senders_graph_i)
        n_edge_list.append(num_edges_graph_i)

        receivers_list.append(receivers_graph_i + num_nodes_in_previous_graphs)
        senders_list.append(senders_graph_i + num_nodes_in_previous_graphs)

        num_nodes_graph_i = len(positions_graph_i)
        num_nodes_in_previous_graphs += num_nodes_graph_i

    # Concatenate all of the results.
    senders = torch.tensor(np.concatenate(senders_list, axis=0).astype(np.int32), dtype=torch.int64, device=device)
    receivers = torch.tensor(np.concatenate(receivers_list, axis=0).astype(np.int32), dtype=torch.int64, device=device)
    n_edge = torch.tensor(np.stack(n_edge_list).astype(np.int32), dtype=torch.int64, device=device)

    return senders, receivers, n_edge
