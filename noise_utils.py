import torch
from learned_simulator import *

def get_random_walk_noise_for_position_sequence(
    position_sequence, noise_std_last_step):
    """Returns random-walk noise in the velocity applied to the position."""
    velocity_sequence = time_diff(position_sequence)

    num_velocities = velocity_sequence.shape[1]
    # We want the noise scale in the velocity at the last step to be fixed.
    # std_each_step `std_last_step / np.sqrt(num_input_velocities)`
    velocity_sequence_noise = torch.normal(
        mean = 0.,
        std = noise_std_last_step / num_velocities ** 0.5,
        size = velocity_sequence.shape,
        dtype = position_sequence.dtype
    )

    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    # Integrate the noise in the velocity to the positions.
    position_sequence_noise = torch.cat([
        torch.zeros_like(velocity_sequence_noise[:, 0:1]),
        torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)

    return position_sequence_noise
