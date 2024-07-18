import const
import housebuilder
import agents.dumb_agent
import agents.pid_agent
import random
import numpy as np
import matplotlib.pyplot as plt

room_air_mass = const.ROOM_LENGTH * const.ROOM_WIDTH * const.ROOM_HEIGHT * const.AIR_DENSITY
wall_area_sum = 2 * (const.ROOM_LENGTH * const.ROOM_HEIGHT + const.ROOM_WIDTH * const.ROOM_HEIGHT)
roof_area = (const.ROOM_LENGTH * const.ROOM_WIDTH)
cool_energy_transfer_watt = -const.COOL_BTUS / 3.41
heat_energy_transfer_watt = const.HEAT_BTUS / 3.41

# power_transfers = [cool_energy_transfer_watt, heat_energy_transfer_watt]

def clamp(val: float, min: float, max: float) -> float:
	if val < min:
		return min
	if val > max:
		return max
	return val

def joule_to_temp_air(joule: float) -> float:
	return joule / (room_air_mass * const.AIR_HEAT_CAPACITY)

# def joule_to_temp_air(joule: float) -> float:
# 	return joule / (room_air_mass * const.AIR_HEAT_CAPACITY)

# convection heat transfer equation Q = hA(Delta)T

# calculate convection from outside air to outer wall
def calc_convection_to_ext_wall(out_wall_temp: float, time: float) -> float:
	change = const.OUTSIDE_CONVECTION_COEFF * wall_area_sum * (const.OUTSIDE_TEMP[time] - out_wall_temp)
	return change * 60

# calculate conduction transfer from outside to inside of wall
def calc_wall_conduction(int_wall_temp: float, out_wall_temp: float) -> float:
	# TODO possible investigate switching int_wall_temp and out_wall_temp
	change = wall_area_sum * (out_wall_temp - int_wall_temp) * const.EXT_WALL_THERM_COND / const.EXT_WALL_THICK
	return change * 60

# calculate convection from inner wall to room air
def calc_wall_convection_to_room(room_temp: float, int_wall_temp: float) -> float:
	change = const.INSIDE_CONVECTION_COEFF * wall_area_sum * (int_wall_temp - room_temp)
	return change * 60

# calculate convection from outside air to roof outside
def calc_convection_to_ext_roof(out_wall_temp: float, time: float) -> float:
	change = const.OUTSIDE_CONVECTION_COEFF * roof_area * (const.OUTSIDE_TEMP[time] - out_wall_temp)
	return change * 60

# calculate conduction transfer from outside to inside of roof
def calc_roof_conduction(int_wall_temp: float, out_wall_temp: float) -> float:
	# TODO possible investigate switching int_wall_temp and out_wall_temp
	change = roof_area * (out_wall_temp - int_wall_temp) * const.ROOF_THERM_COND / const.EXT_WALL_THICK
	return change * 60

# calculate convection from inside of roof to room air
def calc_roof_convection_to_room(room_temp: float, int_wall_temp: float) -> float:
	change = const.INSIDE_CONVECTION_COEFF * roof_area * (int_wall_temp - room_temp)
	return change * 60

# -1 <= power <= 1
# -1 = full cool, 1 = full heat
def calc_ac_effect(power: float) -> float:
	if power == 0:
		return 0
	change = cool_energy_transfer_watt if power < 0 else heat_energy_transfer_watt
	noise = 1 if const.DETERMINISTIC else random.uniform(const.NOISE_MULT_MIN, const.NOISE_MULT_MAX)
	return change * noise * 60

if __name__ == "__main__":
	house = housebuilder.build_house("2r_simple.json")
	rooms = house.get_rooms(0)
	for i in range(len(rooms)):
		print(f"room {i}: {rooms[i]}")
		print(f"room {i} external perimeter: {house.get_external_perimeter(0, i)}")
		print(f"room {i} internal walls: {house.get_internal_walls(0, i)}")
