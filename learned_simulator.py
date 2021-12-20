import torch
import torch.nn as nn
from connectivity_utils import *
from graph_network import EncodeProcessDecode
from dataloader import NCDataset

STD_EPSILON = 1e-8
INPUT_SEQUENCE_LENGTH = 6

def time_diff(input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]

class LearnedSimulator(nn.Module):
    def __init__(self,
                 num_dimensions,
                 connectivity_radius,
                 graph_network_kwargs,
                 boundaries,
                 normalization_stats,
                 num_particle_types,
                 device,
                 particle_type_embedding_size,
                 args,
                 name="LearnedSimulator"):
        """Inits the model.
        Args:
            num_dimensions: Dimensions of the problem.
            connectivity_radius: Scalar with the radius of connectivity.
            graph_network_kwargs: Keyword arguments to pass the learned part of `EncodeProcessDecode`.
            boundaries: List of 2-tuples, containing the lower and upper boundaries of the cuboid containing
            the particles along each dimensions, matching the dimensionality of the problem.
            normalization_stats: Dictionary with statistics with keys "acceleration" and "velocity", containing
            a named tuple for each with mean and std fields, matching the dimensionality of the problem.
            num_particle_types: Number of different particle types.
            particle_type_embedding_size: Embedding size for the particle type.
        """
        super().__init__()

        self._connectivity_radius = connectivity_radius
        self._num_particle_types = num_particle_types
        self._boundaries = boundaries
        self._normalization_stats = normalization_stats
        self._node_input_size = (INPUT_SEQUENCE_LENGTH+1) * num_dimensions
        self._edge_input_size = num_dimensions + 1

        if self._num_particle_types > 1:
            self._particle_type_embedding = nn.Parameter(torch.FloatTensor(self._num_particle_types, particle_type_embedding_size), requires_grad=True)
            self._node_input_size += particle_type_embedding_size

        self._graph_network = EncodeProcessDecode(node_input_size=self._node_input_size, edge_input_size=self._edge_input_size,
                                                  output_size=num_dimensions, device=device, args=args, **graph_network_kwargs).to(device)

    def forward(self, position_sequence, n_particles_per_example,
                global_context=None, particle_types=None):
        input_graphs_tuple = self._encoder_preprocessor(
            position_sequence, n_particles_per_example, global_context,
            particle_types)

        normalized_acceleration = self._graph_network(input_graphs_tuple)

        next_position = self._decoder_postprocessor(
            normalized_acceleration, position_sequence)

        return next_position

    def _encoder_preprocessor(self, position_sequence, n_node, global_context, particle_types):
        # Extract important features from the position_sequence.
        most_recent_position = position_sequence[:, -1] # [num_particles, num_steps, dim]
        velocity_sequence = time_diff(position_sequence)

        senders, receivers, n_edge = compute_connectivity_for_batch(most_recent_position.cpu().numpy(),
                                                                    n_node.cpu().numpy(),
                                                                    self._connectivity_radius,
                                                                    velocity_sequence.device)

        node_features = [] # collect node features

        velocity_stats = self._normalization_stats['velocity']
        velocity_mean = torch.tensor(velocity_stats.mean, device=velocity_sequence.device)
        velocity_std = torch.tensor(velocity_stats.std, device=velocity_sequence.device)
        normalized_velocity_sequence = (velocity_sequence - velocity_mean) / velocity_std
        node_features.append(normalized_velocity_sequence.flatten(1,2)) # merging spatial and time axis, [num_particles, num_steps * dim]

        boundaries = torch.tensor(self._boundaries, dtype=torch.float32, device=most_recent_position.device) # [num_dimensions, 2]
        distance_to_lower_boundary = most_recent_position - torch.unsqueeze(boundaries[:, 0], 0)
        distance_to_upper_boundary = torch.unsqueeze(boundaries[:, 1], 0) - most_recent_position
        distance_to_boundaries = torch.cat([distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
        normalized_clipped_distance_to_boundaries = torch.clip(distance_to_boundaries / self._connectivity_radius, -1., 1.)
        node_features.append(normalized_clipped_distance_to_boundaries)

        if self._num_particle_types > 1:
            particle_type_embeddings = self._particle_type_embedding[particle_types]
            node_features.append(particle_type_embeddings.to(most_recent_position.device))

        edge_features = [] # collect edge features.
        # Relative displacement and distances normalized to radius
        normalized_relative_displacements = (
            most_recent_position[senders] - most_recent_position[receivers]) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = torch.norm(normalized_relative_displacements, dim=-1, keepdim=True)
        edge_features.append(normalized_relative_distances)

        # Normalize the global context.
        if global_context is not None:
            context_stats = self._normalization_stats["context"]
            global_context = (global_context - context_stats.mean) / max(context_stats.std, STD_EPSILON)

        graph_tuple = NCDataset("input_graphs")
        graph_tuple.graph = {
            'node_feat': torch.cat(node_features, dim=-1),
            'edge_feat': torch.cat(edge_features, dim=-1),
            'global': global_context,
            'n_node': n_node,
            'n_edge': n_edge,
            'edge_index': torch.stack([senders, receivers])
        }

        return graph_tuple

    def _decoder_postprocessor(self, normalized_acceleration, position_sequence):
        # apply inverse normalization
        acceleration_stats = self._normalization_stats["acceleration"]
        acceleration_mean = torch.tensor(acceleration_stats.mean, device=normalized_acceleration.device)
        acceleration_std = torch.tensor(acceleration_stats.std, device=normalized_acceleration.device)
        acceleration = (normalized_acceleration * acceleration_std) + acceleration_mean

        # Use an Euler integrator to go from acceleration to position
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = most_recent_position - position_sequence[:, -2]

        new_velocity = most_recent_velocity + acceleration # *dt = 1
        new_position = most_recent_position + new_velocity # *dt = 1
        return new_position

    def _inverse_decoder_postprocessor(self, next_position, position_sequence):
        previous_position = position_sequence[:, -1]
        previous_velocity = previous_position - position_sequence[:, -2]
        next_velocity = next_position - previous_position
        acceleration = next_velocity - previous_velocity

        acceleration_stats = self._normalization_stats['acceleration']
        acceleration_mean = torch.tensor(acceleration_stats.mean, device=acceleration.device)
        acceleration_std = torch.tensor(acceleration_stats.std, device=acceleration.device)
        normalized_acceleration = (acceleration - acceleration_mean) / acceleration_std
        return normalized_acceleration

    def get_predicted_and_target_normalized_accelerations(
        self, next_position, position_sequence_noise, position_sequence,
        n_particles_per_example, global_context=None, particle_types=None):
        noisy_position_sequence = position_sequence + position_sequence_noise

        input_graphs_tuple = self._encoder_preprocessor(noisy_position_sequence,
                            n_particles_per_example, global_context, particle_types)
        predicted_normalized_acceleration = self._graph_network(input_graphs_tuple)

        next_position_adjusted = next_position + position_sequence_noise[:, -1]
        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_position_adjusted, noisy_position_sequence)

        return predicted_normalized_acceleration, target_normalized_acceleration




