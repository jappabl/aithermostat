import matplotlib.pyplot as plt
import numpy as np
import agents.dumb_agent
import const
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import agents.pid_agent
import gym_environment
import sys

env = gym_environment.Environment()

action_size = env.action_space.n
state, _ = env.reset()
observation_size = len(state)

class DQN(nn.Module):
	def __init__(self, observation_size, action_size):
		super().__init__()
		self.fc1 = nn.Linear(observation_size, 16)
		self.fc2 = nn.Linear(16, 32)
		self.fc3 = nn.Linear(32, 64)
		self.fc4 = nn.Linear(64, action_size)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		return self.fc4(x)

policy_net = DQN(observation_size, action_size).to(const.DEVICE)
policy_net.load_state_dict(torch.load("subtract_4_3k_random_weather_2.pt"))

fig, axes = plt.subplots(1, 2)

episode_count = int(sys.argv[-1])
sim_max = 2880
num_setpoints = 4

deviations = {
	"deviation (dumb)": np.zeros(episode_count),
	"deviation (rl)": np.zeros(episode_count),
}
cycles = {
	"cycles (dumb)": np.zeros(episode_count),
	"cycles (rl)": np.zeros(episode_count),
}

for i in range(episode_count):
	deviation_sum = 0
	# weather_start = 0
	weather_start = random.randrange(0, len(const.OUTSIDE_TEMP) - sim_max)
	state, _ = env.reset(num_setpoints=num_setpoints, length=sim_max, start_time=weather_start)
	old_action = 0
	cycles_num = 0
	for t in range(sim_max):	
		power = policy_net(state).max(0).indices.view(1, 1).item()
		if power != old_action:
			cycles_num += 1
		old_action = power

		deviation_sum += abs(env.get_cur_temp() - env.get_setpoint())
		state, reward, _ = env.step(power)
	deviations["deviation (rl)"][i] = deviation_sum / sim_max
	cycles["cycles (rl)"][i] = cycles_num
	print(f"{i + 1}/{episode_count}", end="\r")

print("             ", end="\r")

for i in range(episode_count):
	deviation_sum = 0
	# weather_start = 0
	weather_start = random.randrange(0, len(const.OUTSIDE_TEMP) - sim_max)
	state, _ = env.reset(num_setpoints=num_setpoints, length=sim_max, start_time=weather_start)
	old_action = 0
	cycles_num = 0
	for t in range(sim_max):	
		power = agents.dumb_agent.agent(env.get_cur_temp(), const.OUTSIDE_TEMP[weather_start + t], env.get_setpoint(), env._actions[old_action])
		power = env._actions.index(power)
		if power != old_action:
			cycles_num += 1
		old_action = power

		deviation_sum += abs(env.get_cur_temp() - env.get_setpoint())
		state, reward, _ = env.step(power)
	deviations["deviation (dumb)"][i] = deviation_sum / sim_max
	cycles["cycles (dumb)"][i] = cycles_num
	print(f"{i + 1}/{episode_count}", end="\r")

axes[0].boxplot(deviations.values())
axes[0].set_xticklabels(deviations.keys())

axes[1].boxplot(cycles.values())
axes[1].set_xticklabels(cycles.keys())

fig.set_size_inches(12.8, 9.6)
plt.savefig("out.png", dpi=1000)
