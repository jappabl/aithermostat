import json
import typing
import sys
import numpy as np
import random
import math

# https://en.wikipedia.org/wiki/Machine_epsilon#Values_for_standard_hardware_arithmetics
MACHINE_EPS = 2.23e-16
FILE_NAME = "2r_simple.json"

class Constants(typing.NamedTuple):
	cooler_btu: float
	heater_btu: float

	ext_wall_thick: float
	int_wall_thick: float
	ext_wall_therm_cond: float
	int_wall_therm_cond: float

	ext_roof_thick: float
	int_roof_thick: float
	ext_roof_therm_cond: float
	int_roof_therm_cond: float

	outside_convection: float
	inside_convection: float

	settings: list

	air_density: float
	air_heat_capacity: float

	ext_wall_ext_thick: float
	ext_wall_ext_heat_capacity: float
	ext_wall_ext_density: float
	ext_wall_int_thick: float
	ext_wall_int_heat_capacity: float
	ext_wall_int_density: float

	ext_roof_ext_thick: float
	ext_roof_ext_heat_capacity: float
	ext_roof_ext_density: float
	ext_roof_int_thick: float
	ext_roof_int_heat_capacity: float
	ext_roof_int_density: float

	int_wall_outside_thick: float
	int_wall_outside_heat_capacity: float
	int_wall_outside_density: float

class Opening(typing.NamedTuple):
	start: float
	end: float
	open: bool

class Point(typing.NamedTuple):
	x: float
	y: float
	def get_distance(self, other) -> float:
		return math.sqrt((other.x - self.x) ** 2 + (other.y - self.y) ** 2)
	def __str__(self):
		return f"({self.x}, {self.y})"
	
class Wall:
	def __init__(self, p0: Point, p1: Point):
		self.p0 = p0
		self.p1 = p1

	# given that a point lies on the line L:p0--p1 if it extended infinitely,
	# and that p0.x < p0.y, check whether it actually lies on L
	def is_inside(self, point: Point) -> bool:
		return self.p0.x - MACHINE_EPS <= point.x <= self.p1.x + MACHINE_EPS
	# same as is_inside but wall is vertical
	def is_inside_y(self, point: Point) -> bool:
		y0, y1 = self.p0.y, self.p1.y
		if y0 > y1:	y0, y1 = y1, y0
		return y0 - MACHINE_EPS <= point.y <= y1 + MACHINE_EPS
	
	def length(self) -> float:
		return self.p0.get_distance(self.p1)
	
	def __repr__(self):
		return f"{self.p0}--{self.p1}"

class Room:
	def __init__(self, walls: list, floor_area: float, height: float):
		self.walls = walls
		self.floor_area = floor_area
		self.height = height
		self.volume = floor_area * height
		self.perimeter = 0
		for wall in walls:
			self.perimeter += wall.length()
		self.air_temp = random.uniform(20, 28)

	def joule_to_temp_air(self, joule: float) -> float:
		return joule / (self.volume * const.air_density * const.air_heat_capacity)
	
	def calc_int_convection_to_wall(self, int_wall_temp: float):
		surface_area = self.height * self.perimeter
		wall_outside_mass = surface_area * const.int_wall_outside_thick * const.int_wall_outside_density
		change = const.inside_convection * surface_area * (self.air_temp - int_wall_temp)
		change = change * 60 / (wall_outside_mass * const.int_wall_outside_heat_capacity)
		self.int_wall_temp = int_wall_temp + change
	def calc_int_convection_to_room(self, area: float):
		change = const.inside_convection * area * (self.int_wall_temp - self.air_temp)
		change = change * 60 / (self.volume * const.air_density * const.air_heat_capacity)
		self.air_temp += change
	
	def get_temp(self) -> float:
		return self.air_temp
	def get_volume(self) -> float:
		return self.volume
	# if called before calc_int_convection_to_wall this will fail
	def get_int_wall_temp(self) -> float:
		return self.int_wall_temp
	
	def set_int_wall_temp(self, int_wall_temp: float):
		self.int_wall_temp = int_wall_temp
	def add_air_temp(self, amount: float):
		self.air_temp += amount
	
	def __repr__(self):
		return str(self.__dict__)
	
class House:
	def __init__(self, floors: int):
		self.rooms = [[] for _ in range(floors)]
		self.internal_walls = [[] for _ in range(floors)]
		self.external_perimeter = [[] for _ in range(floors)]
		self.height = [0 for _ in range(floors)]

		self.int_wall_temp = self.int_roof_temp = random.uniform(20, 28)
		self.ext_wall_temp = self.ext_roof_temp = random.uniform(20, 28)

		self.total_external_perimeter = 0
		self.total_roof_area = 0

	def add_room(self, room: Room, floor: int) -> None:
		self.rooms[floor].append(room)
		self.internal_walls[floor].append([])
		self.external_perimeter[floor].append(room.perimeter)
		self.total_external_perimeter += room.perimeter
		self.total_roof_area += room.floor_area

	def add_internal_wall(self, floor: int, room0: int, room1: int, length: float) -> None:
		room0_present, room1_present = -1, -1
		for i in range(len(self.internal_walls[floor][room0])):
			if self.internal_walls[floor][room0][i][0] == room1:
				room0_present = i
		for i in range(len(self.internal_walls[floor][room1])):
			if self.internal_walls[floor][room1][i][0] == room0:
				room1_present = i

		if room0_present == -1:
			self.internal_walls[floor][room0].append((room1, length))
		else:
			self.internal_walls[floor][room0][room0_present][1] += length

		if room1_present == -1:
			self.internal_walls[floor][room1].append((room0, length))
		else:
			self.internal_walls[floor][room1][room1_present][1] += length

		self.external_perimeter[floor][room0] -= length
		self.external_perimeter[floor][room1] -= length
		self.total_external_perimeter -= length * 2

	def set_height(self, floor: int, height: float):
		self.height[floor] = height
	
	def get_rooms(self, floor: int) -> list:
		return self.rooms[floor]
	def get_internal_walls(self, floor: int, room: int) -> list:
		return self.internal_walls[floor][room]
	def get_external_perimeter(self, floor: int, room: int) -> float:
		return self.external_perimeter[floor][room]
	
	def calc_weather_convection_wall(self, outside_temp: float) -> None:
		surface_area = sum(self.height) * self.total_external_perimeter
		wall_outside_mass = surface_area * const.ext_wall_ext_thick * const.ext_wall_ext_density
		change = const.outside_convection * surface_area * (outside_temp - self.ext_wall_temp)
		change = change * 60 / (wall_outside_mass * const.ext_wall_ext_heat_capacity)
		self.ext_wall_temp += change
	def calc_weather_convection_roof(self, outside_temp: float) -> None:
		roof_outside_mass = self.total_roof_area * const.ext_roof_ext_thick * const.ext_roof_ext_density
		change = const.outside_convection * self.total_roof_area * (outside_temp - self.ext_roof_temp)
		change = change * 60 / (roof_outside_mass * const.ext_roof_ext_heat_capacity)
		self.ext_roof_temp += change
	def calc_ext_wall_conduction(self):
		surface_area = sum(self.height) * self.total_external_perimeter
		wall_inside_mass = surface_area * const.ext_wall_int_thick * const.ext_wall_int_density
		wall_outside_mass = surface_area * const.ext_wall_ext_thick * const.ext_wall_ext_density
		change = surface_area * (self.ext_wall_temp - self.int_wall_temp) * const.ext_wall_therm_cond / const.ext_wall_thick
		change_int = change * 60 / (wall_inside_mass * const.ext_wall_int_heat_capacity)
		change_ext = change * 60 / (wall_outside_mass * const.ext_wall_ext_heat_capacity)
		self.int_wall_temp += change_int
		self.ext_wall_temp -= change_ext
	def calc_roof_conduction(self):
		roof_inside_mass = self.total_roof_area * const.ext_roof_int_thick * const.ext_roof_int_density
		roof_outside_mass = self.total_roof_area * const.ext_roof_ext_thick * const.ext_roof_ext_density
		change = self.total_roof_area * (self.ext_roof_temp - self.int_roof_temp) * const.ext_roof_therm_cond / const.ext_roof_thick
		change_int = change * 60 / (roof_inside_mass * const.ext_roof_int_heat_capacity)
		change_ext = change * 60 / (roof_outside_mass * const.ext_roof_ext_heat_capacity)
		self.int_roof_temp += change_int
		self.ext_roof_temp -= change_ext
	def calc_ext_convection_to_room(self, floor: int, room_num: int):
		room: Room = self.rooms[floor][room_num]
		# only external facing walls
		surface_area = sum(self.height) * self.external_perimeter[floor][room_num]
		wall_inside_mass = surface_area * const.ext_wall_int_thick * const.ext_wall_int_density
		roof_inside_mass = self.total_roof_area * const.ext_roof_int_thick * const.ext_roof_int_density

		change_wall = const.inside_convection * surface_area * (self.int_wall_temp - room.get_temp())
		change_roof = const.inside_convection * self.total_roof_area * (self.int_roof_temp - room.get_temp())
		
		change_wall_room = change_wall * 60 / (room.get_volume() * const.air_density * const.air_heat_capacity)
		change_wall_int = change_wall * 60 / (wall_inside_mass * const.ext_wall_ext_heat_capacity)

		change_roof_room = change_roof * 60 / (room.get_volume() * const.air_density * const.air_heat_capacity)
		change_roof_int = change_roof * 60 / (roof_inside_mass * const.ext_roof_ext_heat_capacity)

		room.add_air_temp(change_wall_room + change_roof_room)
		self.int_wall_temp -= change_wall_int
		self.int_roof_temp -= change_roof_int	

	# modifies room0 and room1 with new interior wall temperature
	def calc_conduction_between_rooms(self, room0: Room, room1: Room, area: float):
		wall_outside_mass = area * const.int_wall_outside_thick * const.int_wall_outside_density
		change = area * (room0.get_int_wall_temp() - room1.get_int_wall_temp()) * \
					const.int_wall_therm_cond / const.int_wall_thick
		change = change * 60 / (wall_outside_mass * const.int_wall_outside_heat_capacity)

		room0.set_int_wall_temp(room0.get_int_wall_temp() - change)
		room1.set_int_wall_temp(room1.get_int_wall_temp() + change)

	def step(self, outside_temp: float):
		prev_int_wall_temp = self.int_wall_temp

		self.calc_weather_convection_wall(outside_temp)
		self.calc_weather_convection_roof(outside_temp)
		self.calc_ext_wall_conduction()
		self.calc_roof_conduction()

		for i in range(len(self.rooms)):
			for j in range(len(self.rooms[i])):
				self.calc_ext_convection_to_room(i, j)

		for i in range(len(self.rooms)):
			for j in range(len(self.rooms[i])):
				self.rooms[i][j].calc_int_convection_to_wall(prev_int_wall_temp)
		for i in range(len(self.internal_walls)):
			room_count = len(self.internal_walls[i])
			searched = [[False for _ in range(room_count)] for _ in range(room_count)]
			for j in range(room_count):
				for wall in self.internal_walls[i][j]:
					if j == wall[0]:
						error("jayden li programming error (3)")
					if searched[j][wall[0]] or searched[wall[0]][j]:
						continue
					searched[j][wall[0]] = True
					searched[wall[0]][j] = True

					room0: Room = self.rooms[i][j]
					room1: Room = self.rooms[i][wall[0]]
					wall_area = wall[1] * self.height[i]

					self.calc_conduction_between_rooms(room0, room1, wall_area)
					room0.calc_int_convection_to_room(wall[1])
					room1.calc_int_convection_to_room(wall[1])

	def __repr__(self):
		return str(self.__dict__)

def error(msg: str):
	print("error:", msg, file=sys.stderr)
	sys.exit(1)

def fp_equal(a: float, b: float):
	return a - MACHINE_EPS <= b <= a + MACHINE_EPS 

# returns overlapping length between two walls
def find_overlap(w0: Wall, w1: Wall) -> float:
	# wall points must be sorted by x
	if w0.p0.x > w0.p1.x: w0.p0, w0.p1 = w0.p1, w0.p0
	if w1.p0.x > w1.p1.x: w1.p0, w1.p1 = w1.p1, w1.p0

	w0_vertical = fp_equal(w0.p0.x, w0.p1.x)
	w1_vertical = fp_equal(w1.p0.x, w1.p1.x)
	if w0_vertical and w1_vertical:
		# Case 0. Not all x-coords are the same <=> not lie on same line
		if not w0.p0.x == w0.p1.x == w1.p0.x == w1.p1.x:
			return 0
		
		if w0.p0.y > w0.p1.y: w0.p0, w0.p1 = w0.p1, w0.p0
		if w1.p0.y > w1.p1.y: w1.p0, w1.p1 = w1.p1, w1.p0

		# Case 1. Wall 0 is entirely within Wall 1 (or the walls are the same)
		if w1.is_inside_y(w0.p0) and w1.is_inside_y(w0.p1):
			return w0.p0.get_distance(w0.p1)
		# Case 2. Wall 1 is entirely within Wall 0
		if w0.is_inside_y(w1.p0) and w0.is_inside_y(w1.p1):
			return w1.p0.get_distance(w1.p1)
		# Case 3. Partially overlapping section
		# Subcase 3a. Wall 0 is below Wall 1
		if w0.is_inside_y(w1.p0) and w1.is_inside_y(w0.p1):
			return w1.p0.get_distance(w0.p1)
		# Subcase 3b. Wall 1 is below Wall 0
		if w1.is_inside_y(w0.p0) and w0.is_inside_y(w1.p1):
			return w0.p0.get_distance(w1.p1)
		
		# this shouldn't happen
		error("jayden li programming error (1)")

	if w0_vertical or w1_vertical:
		return 0
	
	slope0 = (w0.p1.y - w0.p0.y) / (w0.p1.x - w0.p0.x)
	slope1 = (w1.p1.y - w1.p0.y) / (w1.p1.x - w1.p0.x)
	if not fp_equal(slope0, slope1):
		return 0

	const0 = w0.p0.y - slope0 * w0.p0.x
	const1 = w1.p0.y - slope1 * w1.p0.x

	# Case 0. Walls are not intersecting
	if not fp_equal(const0, const1):
		return 0

	# Case 1. Wall 0 is entirely within Wall 1 (or the walls are the same)
	if w1.is_inside(w0.p0) and w1.is_inside(w0.p1):
		return w0.p0.get_distance(w0.p1)
	# Case 2. Wall 1 is entirely within Wall 0
	if w0.is_inside(w1.p0) and w0.is_inside(w1.p1):
		return w1.p0.get_distance(w1.p1)
	# Case 3. Partially overlapping section
	# Subcase 3a. Wall 0 is to the left of Wall 1
	if w0.is_inside(w1.p0) and w1.is_inside(w0.p1):
		return w1.p0.get_distance(w0.p1)
	# Subcase 3b. Wall 1 is to the left of Wall 0
	if w1.is_inside(w0.p0) and w0.is_inside(w1.p1):
		return w0.p0.get_distance(w1.p1)
	
	# also shouldn't happen
	error("jayden li programming error (1)")

def build_house(file_name: str) -> House:
	with open(file_name, "r") as file:
		cfg = json.load(file)

	global const
	const = Constants(**cfg["constants"])
	
	floors = cfg["floors"]

	if len(floors) < 1:
		error("floors must have one or more entries", file=sys.stderr)
	if len(floors) > 1:
		print("warning: only 1 floor is currently supported", file=sys.stderr)

	house = House(len(floors))
	for i in range(len(floors)):
		floor = floors[i]
		room_height = floor["height"]
		house.set_height(i, room_height)
		for room_data in floor["rooms"]:
			walls = []
			coords = []

			for wall_data in room_data["walls"]:
				wall = Wall(
					Point(wall_data["x0"], wall_data["y0"]),
					Point(wall_data["x1"], wall_data["y1"])
				)
				windows = wall_data["windows"]
				doors = wall_data["doors"]
				if len(windows) != 0:
					print("warning: windows are not currently supported")
				if len(doors) != 0:
					print("warning: doors are not currently supported")
				coords.append(wall.p0)
				coords.append(wall.p1)
				walls.append(wall)

			new_coords = []
			for coord in coords:
				if coords.count(coord) != 2:
					error("illegal room")
				if coord not in new_coords:
					new_coords.append(coord)
					
			# https://en.wikipedia.org/wiki/Shoelace_formula#Trapezoid_formula
			area = 0
			new_coords.append(new_coords[0])
			for j in range(len(new_coords) - 1):
				area += (new_coords[j].y + new_coords[j + 1].y) * (new_coords[j].x - new_coords[j + 1].x)
			area /= 2
			# print(area)

			room = Room(walls, area, room_height)
			floor_room_list = house.get_rooms(i)
			internal_walls = []

			for other_room_index in range(len(house.get_rooms(i))):
				other_room = floor_room_list[other_room_index]
				for other_wall in other_room.walls:
					for wall in walls:
						overlap = find_overlap(wall, other_wall)
						if overlap == 0:
							continue
						internal_walls.append((other_room_index, overlap))

			cur_room_num = len(house.get_rooms(i))
			house.add_room(room, i)
			for other_room_index, length in internal_walls:
				house.add_internal_wall(i, cur_room_num, other_room_index, length)

		# only support one floor (i am lazy)
		break

	return house
