import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import const

import gym_environment
env = gym_environment.Environment()

import matplotlib.pyplot as plt
import numpy as np
import const
import random
import time
import agents.dumb_agent

fig, ax1 = plt.subplots()
ax1.set_xlabel("time (min)")
# ax1.set_ylim(10, 30)
# ax1.set_yticks(np.arange(10, 31))
ax1.set_ylabel("deg C")

sim_max = 2880

# weather_start = 695991
weather_start = random.randrange(0, len(const.OUTSIDE_TEMP) - sim_max)

xvalues = np.arange(0, sim_max)
temperatures = np.zeros(sim_max)
outside_temp = np.zeros(sim_max)

state, _ = env.reset(num_setpoints=1, length=sim_max, start_time=weather_start)

for i in range(sim_max):	
	temperatures[i] = env.get_cur_temp()
	outside_temp[i] = const.OUTSIDE_TEMP[weather_start + i]
	state, _, _ = env.step(1)

ax1.plot(xvalues, temperatures, color="red", linewidth=0.1)
ax1.plot(xvalues, outside_temp, color="green", linewidth=0.1)
# plt.show()
plt.savefig("old2.png", dpi=1000)

# cycles
# m2K/W * m2 * K
