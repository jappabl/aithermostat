import json
import typing
import sys
import numpy as np
import math

# https://en.wikipedia.org/wiki/Machine_epsilon#Values_for_standard_hardware_arithmetics
MACHINE_EPS = 2.23e-16

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
	def __repr__(self):
		return str(self.__dict__)
class House:
	def __init__(self, floors: int):
		self.rooms = [[] for _ in range(floors)]
		self.internal_walls = [[] for _ in range(floors)]
		self.external_perimeter = [[] for _ in range(floors)]
	def add_room(self, room: Room, floor: int) -> None:
		self.rooms[floor].append(room)
		self.internal_walls[floor].append([])
		self.external_perimeter[floor].append(room.perimeter)
	def add_internal_wall(self, floor: int, room0: int, room1: int, length: float) -> None:
		self.internal_walls[floor][room0].append((room1, length))
		self.internal_walls[floor][room1].append((room0, length))
		self.external_perimeter[floor][room0] -= length
		self.external_perimeter[floor][room1] -= length
	
	def get_rooms(self, floor: int) -> list:
		return self.rooms[floor]
	def get_internal_walls(self, floor: int, room: int) -> list:
		return self.internal_walls[floor][room]
	def get_external_perimeter(self, floor: int, room: int) -> float:
		return self.external_perimeter[floor][room]
	
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

	w0_horizontal = fp_equal(w0.p0.x, w0.p1.x)
	w1_horizontal = fp_equal(w1.p0.x, w1.p1.x)
	if w0_horizontal and w1_horizontal:
		# TODO special case
		return 0
	if w0_horizontal or w1_horizontal:
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
	
	# this shouldn't happen
	error("jayden li programming error")
	
with open("house.json") as file:
	cfg = json.load(file)

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
				error("illegal room (rule 1)")
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

	break

print(house)