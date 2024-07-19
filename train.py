import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import gym_environment
import const

env = gym_environment.Environment()
	
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1
EPS_END = 0
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# from the pytorch page on dqn
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
class ReplayMemory:
	def __init__(self, size):
		self.memory = deque([], maxlen=size)
	def push(self, *args):
		self.memory.append(Transition(*args))
	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)
	def __len__(self):
		return len(self.memory)
	
class DQN(nn.Module):
	def __init__(self, observation_size, action_size):
		super().__init__()
		# self.fc1 = nn.Linear(observation_size, 128)
		# self.fc2 = nn.Linear(128, 128)
		# self.fc3 = nn.Linear(128, action_size)
		self.fc1 = nn.Linear(observation_size, 16)
		self.fc2 = nn.Linear(16, 32)
		self.fc3 = nn.Linear(32, 64)
		self.fc4 = nn.Linear(64, action_size)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return self.fc4(x)

action_size = env.action_space.n
state, _ = env.reset()
observation_size = len(state)

policy_net = DQN(observation_size, action_size).to(const.DEVICE)
target_net = DQN(observation_size, action_size).to(const.DEVICE)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
	global steps_done
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
	steps_done += 1
	if sample > eps_threshold:
		with torch.no_grad():
			# t.max(1) will return the largest column value of each row.
			# second column on max result is index of where max element was
			# found, so we pick action with the larger expected reward.
			return policy_net(state).max(1).indices.view(1, 1)
	else:
		return torch.tensor([[env.action_space.sample()]], device=const.DEVICE, dtype=torch.long)

def optimize_model():
	if len(memory) < BATCH_SIZE:
		return
	
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))
	# print(batch)

	non_final_mask = torch.tensor(list(map(lambda s: s is not None, batch.next_state)), device=const.DEVICE, dtype=torch.bool)
	non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

	state_batch = torch.cat(batch.state)
	action_batch = torch.cat(batch.action)
	reward_batch = torch.cat(batch.reward)

	# Compute Q(s_t, a) - the model computes Q(s_t), then we select the
	# columns of actions taken. These are the actions which would've been taken
	# for each batch state according to policy_net
	state_action_values = policy_net(state_batch).gather(1, action_batch)

	# Compute V(s_{t+1}) for all next states.
	# Expected values of actions for non_final_next_states are computed based
	# on the "older" target_net; selecting their best reward with max(1).values
	# This is merged based on the mask, such that we'll have either the expected
	# state value or 0 in case the state was final.
	next_state_values = torch.zeros(BATCH_SIZE, device=const.DEVICE)
	with torch.no_grad():
		next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
	# Compute the expected Q values
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	# Compute Huber loss
	criterion = nn.SmoothL1Loss()
	loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

	print(loss)

	# Optimize the model
	optimizer.zero_grad()
	loss.backward()
	# In-place gradient clipping
	torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
	optimizer.step()

term_cols = os.get_terminal_size().columns

# if torch.cuda.is_available() or torch.backends.mps.is_available():
# 	num_episodes = 1000
# else:
# 	num_episodes = 1000

num_episodes = int(sys.argv[-1])

xvalues = np.arange(1441)
temps = np.zeros(1441)
target = np.zeros(1441)
reward2 = np.zeros(1441)

for i_episode in range(num_episodes):
	rewards = []
	total_reward = 0
	# Initialize the environment and get its state

	state, info = env.reset(num_setpoints=random.randint(2, 7))
	
	state = state.unsqueeze(0)

	for t in count():
		action = select_action(state)
		observation, reward, terminated = env.step(action.item())

		# temps[t] = env._cur_temp
		# target[t] = env._target
		# reward2[t] = reward

		reward = torch.tensor([reward], device=const.DEVICE)
		done = terminated 

		if terminated:
			next_state = None
		else:
			next_state = observation.unsqueeze(0)

		# Store the transition in memory
		memory.push(state, action, next_state, reward)

		# Move to the next state
		state = next_state

		# Perform one step of the optimization (on the policy network)
		optimize_model()

		# Soft update of the target network's weights
		# θ′ ← τ θ + (1 −τ )θ′
		target_net_state_dict = target_net.state_dict()
		policy_net_state_dict = policy_net.state_dict()
		for key in policy_net_state_dict:
			target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
		target_net.load_state_dict(target_net_state_dict)

		rewards.append(reward.item())
		if done:
			print(f"{' ' * term_cols}\repisode {i_episode} sum {sum(rewards)}", end="\r")
			break

# plt.plot(xvalues, temps, linewidth=0.1)
# plt.plot(xvalues, target, linewidth=0.1)
# plt.plot(xvalues, reward2, linewidth=0.1)
# plt.ioff()
# plt.savefig("out.png", dpi=3000)

torch.save(policy_net.state_dict(), "subtract_4_3k_random_weather_2.pt")