import torch
from utils import INPUT_SEQUENCE_LENGTH, get_kinematic_mask

def rollout(simulator, features, num_steps):
    """Rolls out a trajectory by applying the model in sequence."""
    initial_positions = features['positions'][:, 0:INPUT_SEQUENCE_LENGTH]
    ground_truth_positions = features['positions'][:, INPUT_SEQUENCE_LENGTH:]
    global_context = features.get('step_context')

    def step_fn(step, current_positions, predictions):
        if global_context is None:
            global_context_step = None
        else:
            global_context_step = global_context[step + INPUT_SEQUENCE_LENGTH - 1].unsqueeze(0)

        next_position = simulator(
            current_positions,
            n_particles_per_example=features['n_particles_per_example'],
            particle_types=features['particle_types'],
            global_context=global_context_step)

        kinematic_mask = get_kinematic_mask(features['particle_types']).unsqueeze(1).tile(2)
        next_position_ground_truth = ground_truth_positions[:, step]
        next_position = torch.where(kinematic_mask, next_position_ground_truth, next_position)
        predictions.append(next_position)

        # Shift `current_positions`, removing the oldest position in the sequence
        # and appending the next position at the end
        next_positions = torch.cat([current_positions[:, 1:], next_position.unsqueeze(1)], dim=1)

        return (step + 1, next_positions, predictions)

    step = 0; predictions = []; next_positions = initial_positions
    for _ in range(num_steps):
        (step, next_positions, predictions) = step_fn(step, next_positions, predictions)

    output_dict = {
        'initial_positions': torch.transpose(initial_positions, 0, 1),
        'predicted_rollout': torch.stack(predictions),
        'ground_truth_rollout': torch.transpose(ground_truth_positions, 0, 1),
        'particle_types': features['particle_types'],
    }

    if global_context is not None:
        output_dict['global_context'] = global_context
    return output_dict