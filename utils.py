import numpy as np
from texttable import Texttable
import random
import torch
import collections
import os
import json

def print_args(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 6 # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_kinematic_mask(particle_types):
    return torch.eq(particle_types, KINEMATIC_PARTICLE_ID)

def _combine_std(std_x, std_y):
    return np.sqrt(std_x**2 + std_y**2)

def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())