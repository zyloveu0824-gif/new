import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import datetime
import json
import os
from MEC_env import mec_def
from MEC_env import mec_env
import ppo_torch  # Assuming this is your PyTorch-based agent implementation

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seeds for reproducibility
torch.manual_seed(17)
np.random.seed(1)
random.seed(17)

# Parameters
map_size = 200
agent_num = 6
sensor_num = 30
obs_r = 60
collect_r = 40
speed = 6
max_size = 10
sensor_lam = 1e3 + 10 * 200
MAX_EPOCH = 5000
MAX_EP_STEPS = 200
LR_A = 0.001
LR_C = 0.002
GAMMA = 0.85
TAU = 0.8
BATCH_SIZE = 64
Epsilon = 0.2
up_freq = 8
render_freq = 32
FL = True
FL_omega = 0.5

# Parameters dictionary
params = {
    'map_size': map_size,
    'agent_num': agent_num,
    'sensor_num': sensor_num,
    'obs_r': obs_r,
    'collect_r': collect_r,
    'speed': speed,
    'max_size': max_size,
    'sensor_lam': sensor_lam,
    'MAX_EPOCH': MAX_EPOCH,
    'MAX_EP_STEPS': MAX_EP_STEPS,
    'LR_A': LR_A,
    'LR_C': LR_C,
    'GAMMA': GAMMA,
    'TAU': TAU,
    'BATCH_SIZE': BATCH_SIZE,
    'Epsilon': Epsilon,
    'learning_seed': 17,
    'env_seed': 1,
    'up_freq': up_freq,
    'render_freq': render_freq,
    'FL': FL,
    'FL_omega': FL_omega
}

# Initialize MEC world and environment
mec_world = mec_def.MEC_world(map_size, agent_num, sensor_num, obs_r, speed, collect_r, MAX_EP_STEPS, max_size, sensor_lam)
env = mec_env.MEC_MARL_ENV(mec_world)

# Initialize MAAC agent
MAAC = ppo_torch.MAACAgent(env, TAU, GAMMA, LR_A, LR_C, LR_A, LR_C, LR_A, LR_C, BATCH_SIZE, Epsilon)

# Create directories for logs
log_dirs = ["logs/hyperparam", "logs/env", "logs/fit", "logs/models", "logs/records"]
for directory in log_dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Save parameters to JSON file
m_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
with open(f'logs/hyperparam/{m_time}.json', 'w') as f:
    json.dump(params, f)

# Training loop
MAAC.train(MAX_EPOCH, MAX_EP_STEPS, up_freq=up_freq, render=True, render_freq=render_freq, FL=FL, FL_omega=FL_omega)
