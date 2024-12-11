import math
from math import sqrt
from copy import deepcopy
from typing import Optional
from heapq import heappush, heappop
from random import randrange
import random
import ompl.src.ompl.base as ob
import ompl.src.ompl.geometric as og
import numpy as np
from solutions.drone_modules.occupancy_grid import OccupancyGrid

# Utilisation d'OMPL afin d'obtenir un chemin

class PathPlanner:
    def __init__(self, grid, resolution):
        self.grid = grid                # Initialise la grille avec la occupancy grid                                                                       
        self.resolution = resolution   # Initalise la taille d'une cellule de la grille

        # Initialiser l'espace d'état 2D (x, y)
        space = ob.RealVectorStateSpace(2)      # Représente l'espace
        bounds = ob.RealVectorBounds(2)         # Représente les limites 

        # Définir les limites de l'espace en fonction de la taille de la grille
        bounds.setLow(0)
        bounds.setHigh(0, self.grid.x_max_grid * self.resolution)  # Limite x
        bounds.setHigh(1, self.grid.y_max_grid * self.resolution)  # Limite y
        space.setBounds(bounds)

        self.space = space                                     # Initialisation de l'espace
        self.spaceInformation= ob.SpaceInformation(space)      # Initialisation des informations de l'espace

        # Ajouter un validateur d'état basé sur la grille d'occupation
        self.spaceInformation.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))

    def is_state_valid(self, state):
        x, y = state[0], state[1]
        grid_x = int(x / self.resolution)
        grid_y = int(y / self.resolution)

        # Vérifier si les coordonnées sont dans les limites de la grille
        if 0 <= grid_x < self.grid.x_max_grid and 0 <= grid_y < self.grid.y_max_grid:
            return self.grid.grid[grid_x, grid_y] < 0  # Cellule libre si < 0
        return False

    def plan_path(self, start, goal):
        # Définir les états de départ et d'arrivée
        start_state = ob.State(self.space)
        start_state[0], start_state[1] = start

        # Liste donnant le chemin à suivre
        waypoints = []

        goal_state = ob.State(self.space)
        goal_state[0], goal_state[1] = goal

        # Définir le problème de planification
        problem = ob.ProblemDefinition(self.spaceInformation)
        problem.setStartAndGoalStates(start_state, goal_state)

        # Planification avec RRT
        planner = og.RRT(self.spaceInformation)
        planner.setProblemDefinition(problem)
        planner.setup()

        # Recherche d'une solution dans un temps limite de 10 secondes
        solved = planner.solve(10.0)

        # Récupération des points afin de créer le chemin
        if solved:
            path = problem.getSolutionPath()
            for state in path.getStates():
                waypoints.append((state[0],state[1]))
        return waypoints

    
