import os
import warnings
warnings.filterwarnings("ignore")
from torch import nn
import torch
import gym
from collections import deque
import itertools
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter

from baselines_wrappers import DummyVecEnv, SubprocVecEnv, Monitor
from pytorch_wrappers import make_atari_deepmind, BatchedPytorchFrameStack, PytorchLazyFrames

import msgpack
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()

GAME_NAME = 'Breakout-v0'
LOAD_DIR = '/home/twel/CS211_Assignment02/model/[Best model DoubleDQN] [4340000] Breakout-v0.pack'

GAMMA=0.99
BATCH_SIZE=32
BUFFER_SIZE=int(1e6)
MIN_REPLAY_SIZE=50000
EPSILON_START=1.0
EPSILON_END=0.1
EPSILON_DECAY=int(1e6)
NUM_ENVS = 4
TARGET_UPDATE_FREQ=10000 // NUM_ENVS
LR = 5e-5
SAVE_PATH = './breakoutv0_model_double.pack'.format(LR)
SAVE_INTERVAL = 10000
LOG_DIR = './logs/breakoutv0_double' + str(LR)
LOG_INTERVAL = 1000

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


def nature_cnn(observation_space, depths=(32, 64, 64), final_layer=512):
    n_input_channels = observation_space.shape[0]

    cnn = nn.Sequential(
        nn.Conv2d(n_input_channels, depths[0], kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(depths[0], depths[1], kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(depths[1], depths[2], kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten())

    # Compute shape by doing one forward pass
    with torch.no_grad():
        n_flatten = cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

    out = nn.Sequential(cnn, nn.Linear(n_flatten, final_layer), nn.ReLU())

    return out

class Network(nn.Module):
    def __init__(self, env, device, double=True):
        super().__init__()

        self.num_actions = env.action_space.n
        self.device = device
        self.double = double

        conv_net = nature_cnn(env.observation_space)

        self.net = nn.Sequential(conv_net, nn.Linear(512, self.num_actions))

    def forward(self, x):
        return self.net(x)

    def act(self, obses, epsilon):
        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        q_values = self(obses_t)

        max_q_indices = torch.argmax(q_values, dim=1)
        actions = max_q_indices.detach().tolist()

        for i in range(len(actions)):
            rnd_sample = random.random()
            if rnd_sample <= epsilon:
                actions[i] = random.randint(0, self.num_actions - 1)

        return actions

    def compute_loss(self, transitions, target_net):
        obses = [t[0] for t in transitions]
        actions = np.asarray([t[1] for t in transitions])
        rews = np.asarray([t[2] for t in transitions])
        dones = np.asarray([t[3] for t in transitions])
        new_obses = [t[4] for t in transitions]

        if isinstance(obses[0], PytorchLazyFrames):
            obses = np.stack([o.get_frames() for o in obses])
            new_obses = np.stack([o.get_frames() for o in new_obses])
        else:
            obses = np.asarray(obses)
            new_obses = np.asarray(new_obses)

        obses_t = torch.as_tensor(obses, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device).unsqueeze(-1)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32, device=self.device)

        # Compute Targets
        with torch.no_grad():
            if self.double:
                targets_online_q_values = self(new_obses_t)
                targets_online_best_q_indices = targets_online_q_values.argmax(dim=1, keepdim=True)

                targets_target_q_values = target_net(new_obses_t)
                targets_selected_q_values = torch.gather(input=targets_target_q_values, dim=1, index=targets_online_best_q_indices)

                targets = rews_t + GAMMA * (1 - dones_t) * targets_selected_q_values
            else:
                target_q_values = target_net(new_obses_t)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

                targets = rews_t + GAMMA * (1 - dones_t) * max_target_q_values

        # Compute Loss
        q_values = self(obses_t)

        action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

        loss = nn.functional.smooth_l1_loss(action_q_values, targets)

        return loss

    def save(self, save_path):
        params = {k: t.detach().cpu().numpy() for k, t in self.state_dict().items()}
        params_data = msgpack.dumps(params)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            f.write(params_data)

    def load(self, load_path):
        if not os.path.exists(load_path):
            raise FileNotFoundError(load_path)

        with open(load_path, 'rb') as f:
            params_numpy = msgpack.loads(f.read())

        params = {k: torch.as_tensor(v, device=self.device) for k,v in params_numpy.items()}

        self.load_state_dict(params)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print('device:', device)

make_env = lambda: make_atari_deepmind(GAME_NAME, render=True)

vec_env = DummyVecEnv([make_env for _ in range(1)])

env = BatchedPytorchFrameStack(vec_env, k=4)

net = Network(env, device)
net = net.to(device)

net.load(LOAD_DIR)

obs = env.reset()
beginning_episode = True
for t in itertools.count():
    if isinstance(obs[0], PytorchLazyFrames):
        act_obs = np.stack([o.get_frames() for o in obs])
        action = net.act(act_obs, 0.0)
    else:
        action = net.act(obs, 0.0)

    if beginning_episode:
        action = [1]
        beginning_episode = False

    obs, rew, done, _ = env.step(action)
    
    if done[0]:
        obs = env.reset()
        beginning_episode = True