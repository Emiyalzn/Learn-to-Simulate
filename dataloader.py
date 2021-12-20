import torch
import pickle
import os
import numpy as np
from torch.utils.data import Dataset

INPUT_SEQUENCE_LENGTH = 6

class NCDataset(object):
    def __init__(self, name):
        self.name = name
        self.graph = {}
        self.label = None

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

class OneStepDataset(Dataset):
    def __init__(self, dataset='Water', split='train'):
        positions, particle_types, step_context = load_data(dataset, split)
        features = {}
        positions = list(map(torch.tensor, positions))
        particle_types = list(map(torch.tensor, particle_types))
        features['positions'] = positions
        features['particle_types'] = particle_types
        if step_context:
            step_context = list(map(torch.tensor, step_context))
            features['step_context'] = step_context
        self.features = split_trajectory(features, INPUT_SEQUENCE_LENGTH+1)

    def __getitem__(self, index):
        feature = {}
        feature['positions'] = self.features['positions'][index]
        feature['particle_types'] = self.features['particle_types'][index]
        if 'step_context' in self.features:
            feature['step_context'] = self.features['step_context'][index]
        return prepare_inputs(feature)

    def __len__(self):
        return len(self.features['positions'])

class RolloutDataset(Dataset):
    def __init__(self, dataset='Water', split='test'):
        positions, particle_types, step_context = load_data(dataset, split)
        features = {}
        positions = list(map(torch.tensor, positions))
        particle_types = list(map(torch.tensor, particle_types))
        features['positions'] = positions
        features['particle_types'] = particle_types
        if step_context:
            step_context = list(map(torch.tensor, step_context))
            features['step_context'] = step_context
        self.features = features

    def __getitem__(self, index):
        feature = {}
        feature['positions'] = self.features['positions'][index]
        feature['particle_types'] = self.features['particle_types'][index]
        if 'step_context' in self.features:
            feature['step_context'] = self.features['step_context'][index]
        return prepare_rollout_inputs(feature)

    def __len__(self):
        return len(self.features['positions'])

def split_trajectory(features, window_length=7):
    """Splits trajectory into sliding windows."""
    trajectory_length = features['positions'][0].shape[0]
    input_trajectory_length = trajectory_length - window_length + 1
    model_input_features = {
        'particle_types': [],
        'positions': [],
    }
    if 'step_context' in features:
        model_input_features['step_context'] = []
    for i in range(len(features['positions'])):
        # Prepare the context features per step.
        for idx in range(input_trajectory_length):
            model_input_features['positions'].append(features['positions'][i][idx:idx + window_length])
            model_input_features['particle_types'].append(features['particle_types'][i])
            if 'step_context' in features:
                model_input_features['step_context'].append(features['step_context'][i][idx:idx + window_length])

    return model_input_features

def load_data(dataset, split):
    datapath = f"datasets/{dataset}/{split}"
    position_file = os.path.join(datapath, "positions.pkl")
    particle_file = os.path.join(datapath, "particle_types.pkl")
    context_file = os.path.join(datapath, "step_context.pkl")

    with open(position_file, 'rb') as f:
        positions = pickle.load(f)
    with open(particle_file, 'rb') as f:
        particle_types = pickle.load(f)
    if os.path.exists(context_file):
        with open(context_file, 'rb') as f:
            step_context = pickle.load(f)
        return positions, particle_types, step_context

    return positions, particle_types, None

def one_step_collate(batch):
    output_dict = {}
    output_dict['positions'] = torch.cat([sample['positions'] for sample, _ in batch], dim=0)
    output_dict['n_particles_per_example'] = torch.cat([sample['n_particles_per_example'] for sample, _ in batch], dim=0)
    output_dict['particle_types'] = torch.cat([sample['particle_types'] for sample, _ in batch], dim=0)
    sample, _ = batch[0]
    if 'step_context' in sample:
        output_dict['step_context'] = torch.cat([sample['step_context'] for sample, _ in batch], dim=0)
    output_target = torch.cat([sample for _, sample in batch], dim=0)

    return output_dict, output_target

def prepare_inputs(tensor_dict):
    pos = tensor_dict['positions'] # [num_steps, num_particles, num_dims]
    pos = torch.transpose(pos, 0, 1)  # [num_particles, num_steps, num_dims]

    target_position = pos[:, -1]
    tensor_dict['positions'] = pos[:, :-1] # remove the target from the input.

    num_particles = torch.tensor(pos.shape[0])
    tensor_dict['n_particles_per_example'] = num_particles.unsqueeze(0)

    if 'step_context' in tensor_dict:
        tensor_dict['step_context'] = tensor_dict['step_context'][-2] # remove the target from the input.
        tensor_dict['step_context'] = tensor_dict['step_context'].unsqueeze(0)

    return tensor_dict, target_position

def prepare_rollout_inputs(features):
    """Prepares an input trajectory for rollout."""
    out_dict = {}
    out_dict['particle_types'] = features['particle_types']
    pos = torch.transpose(features['positions'], 0, 1) # [num_particles, num_steps, num_dims]

    target_position = pos[:, -1]
    out_dict['positions'] = pos[:, :-1]

    out_dict['n_particles_per_example'] = torch.tensor(pos.shape[0]).unsqueeze(0)

    if 'step_context' in features:
        out_dict['step_context'] = features['step_context']
    out_dict['is_trajectory'] = torch.tensor([True], dtype=torch.bool)
    return out_dict, target_position

