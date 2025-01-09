import math
from math import sqrt
from copy import deepcopy
from typing import Optional
from heapq import heappush, heappop
from random import randrange
from spg_overlay.utils.grid import Grid
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
import random
import numpy as np
from solutions.drone_modules.occupancy_grid import OccupancyGrid
from random import uniform

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0  # Coût pour atteindre ce nœud

class PathPlanner:
    def __init__(self, grid, resolution, size_area_world, drone):
        self.size_area_world = size_area_world
        self.resolution = resolution
        self.drone = drone
        self.grid = grid
        self.x_max_grid = int(self.size_area_world[0] / self.resolution + 0.5)
        self.y_max_grid = int(self.size_area_world[1] / self.resolution + 0.5)
        self.obstacles = []
        self.get_obstacles()

    def get_obstacles(self):
        obstacles = []
        for x in range(self.x_max_grid):
            for y in range(self.y_max_grid):
                if self.grid[x][y] > 0:  # Supposons que self.grid est un tableau 2D
                    obstacles.append((x, y))
        return obstacles


    def is_collision(self, x1, y1, x2, y2):
        """Vérifie s'il existe une collision sur le segment entre deux points."""
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x2), int(y2)
        for t in np.linspace(0, 1, num=100):  # Discrétisation du segment
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            if (x, y) in self.obstacles:
                return True
        return False

    def sample_free(self):
        """Génère un point aléatoire libre dans la zone de la grille."""
        while True:
            x = uniform(0, self.x_max_grid)
            y = uniform(0, self.y_max_grid)
            if (int(x), int(y)) not in self.obstacles:
                return x, y

    def nearest_node(self, tree, point):
        """Trouve le nœud le plus proche d'un point donné."""
        distances = [(node, math.hypot(node.x - point[0], node.y - point[1])) for node in tree]
        return min(distances, key=lambda x: x[1])[0]

    def steer(self, from_node, to_point, max_step_size):
        """Crée un nouveau nœud en direction d'un point donné avec une longueur limitée."""
        theta = math.atan2(to_point[1] - from_node.y, to_point[0] - from_node.x)
        dist = min(max_step_size, math.hypot(to_point[0] - from_node.x, to_point[1] - from_node.y))
        new_x = from_node.x + dist * math.cos(theta)
        new_y = from_node.y + dist * math.sin(theta)
        return Node(new_x, new_y)

    def rewire(self, tree, new_node, radius):
        """Réorganise les parents des nœuds pour minimiser les coûts dans un rayon donné."""
        for node in tree:
            if math.hypot(node.x - new_node.x, node.y - new_node.y) < radius:
                cost = new_node.cost + math.hypot(node.x - new_node.x, node.y - new_node.y)
                if cost < node.cost and not self.is_collision(node.x, node.y, new_node.x, new_node.y):
                    node.parent = new_node
                    node.cost = cost

    def plan(self, start, goal, max_iter=1000, max_step_size=5, radius=10):
        """Implémente l'algorithme RRT*."""
        start_node = Node(start[0], start[1])
        goal_node = Node(goal[0], goal[1])
        tree = [start_node]

        for _ in range(max_iter):
            rand_point = self.sample_free()
            nearest = self.nearest_node(tree, rand_point)
            new_node = self.steer(nearest, rand_point, max_step_size)

            if self.is_collision(nearest.x, nearest.y, new_node.x, new_node.y):
                continue

            new_node.parent = nearest
            new_node.cost = nearest.cost + math.hypot(new_node.x - nearest.x, new_node.y - nearest.y)
            tree.append(new_node)

            self.rewire(tree, new_node, radius)

            if math.hypot(new_node.x - goal_node.x, new_node.y - goal_node.y) < max_step_size:
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + math.hypot(goal_node.x - new_node.x, goal_node.y - new_node.y)
                tree.append(goal_node)
                break

        # Construction du chemin
        path = []
        node = goal_node
        while node.parent is not None:
            path.append((node.x, node.y))
            node = node.parent
        path.append((start_node.x, start_node.y))
        path.reverse()
        return path

# Exemple d'utilisation
# grid = np.zeros((100, 100))  # Grille vide
# planner = PathPlanner(grid, resolution=1, size_area_world=(100, 100), drone=None)
# path = planner.plan(start=(10, 10), goal=(90, 90))
# print(path)

