import numpy as np
import random
import gymnasium as gym
import sim
import torch
import const

class Environment(gym.Env):
	def _sgn(self, num):
		if self._actions[num] < 0: return -1
		if self._actions[num] > 0: return 1
		return 0
	def __init__(self):
		self.observation_space = gym.spaces.Box(0, 4000, shape=(5,), dtype=float)
		self.action_space = gym.spaces.Discrete(3)
		self._actions = [-1, 0, 1]

	def _get_observations(self):
		return torch.tensor([self._cur_temp, self._cur_setpoint, const.OUTSIDE_TEMP[self._time], self._last_toggle, self._old_power], device=const.DEVICE)
	def _get_reward(self):
		return -abs(self._cur_setpoint - self._cur_temp) # lower = better, higher = worse
		# return -math.exp(abs(self._target - self._cur_temp) - 1.7)
	
	def reset(self, num_setpoints=1, length=1440, start_time=None):
		super().reset()

		self._setpoint_list = {}
		self._cur_setpoint = self._setpoint_list[0] = random.uniform(20, 28)
		for _ in range(num_setpoints - 1):
			self._setpoint_list[random.randrange(1, length)] = random.uniform(20, 28)
		
		self._cur_temp = random.uniform(20, 28)
		self._old_power = 1 # TODO check if this makes sense (maybe should be 0???)
		self._last_toggle = -1000
		self._time = 0
		self._length = length
		self._start_time = start_time if start_time is not None else \
							random.randrange(0, len(const.OUTSIDE_TEMP) - self._length)
		
		self._cur_out_wall_temp = const.WALL_STARTING_TEMP
		self._cur_int_wall_temp = self._cur_temp
		self._cur_out_roof_temp = const.WALL_STARTING_TEMP
		self._cur_int_roof_temp = self._cur_temp

		return self._get_observations(), self._get_reward()

	def step(self, power):
		if self._time in self._setpoint_list:
			self._cur_setpoint = self._setpoint_list[self._time]

		self._cur_out_wall_temp += sim.calc_convection_to_ext_wall(
			self._cur_out_wall_temp,
			self._time + self._start_time
		)
		self._cur_int_wall_temp += sim.calc_wall_conduction(
			self._cur_int_wall_temp,
			self._cur_out_wall_temp
		)
		wall_convection_to_room = sim.calc_wall_convection_to_room(
			self._cur_temp,
			self._cur_int_wall_temp
		)
		self._cur_int_wall_temp -= wall_convection_to_room
		self._cur_temp += wall_convection_to_room

		self._cur_out_roof_temp += sim.calc_convection_to_ext_roof(
			self._cur_out_roof_temp,
			self._time + self._start_time
		)
		self._cur_int_roof_temp += sim.calc_roof_conduction(
			self._cur_int_roof_temp,
			self._cur_out_roof_temp
		)
		roof_convection_to_room = sim.calc_roof_convection_to_room(
			self._cur_temp,
			self._cur_int_roof_temp
		)
		self._cur_int_roof_temp -= roof_convection_to_room
		self._cur_temp += roof_convection_to_room

		self._cur_temp += sim.calc_ac_effect(self._actions[power])

		reward = self._get_reward()
		terminated = self._time > self._length

		if self._sgn(self._old_power) != self._sgn(power):
			reward -= max(0, self._last_toggle - self._time + 31.5)

		self._time += 1
		self._old_power = power
		
		return self._get_observations(), reward, terminated
	
	def get_setpoint(self):
		return self._cur_setpoint
	def get_cur_temp(self):
		return self._cur_temp
	
	def calculate_power(self, action):
		return self._actions[action]