import sys
import random
import math
from pathlib import Path
from typing import Optional, List, Type
from enum import Enum
import numpy as np
import cv2

# Insert the parent directory of the current file's directory into sys.path.
# This allows Python to locate modules that are one level above the current
# script, in this case spg_overlay.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
from spg_overlay.utils.grid import Grid
from spg_overlay.utils.pose import Pose
from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.gui_map.gui_sr import GuiSR
from maps.map_intermediate_01 import MyMapIntermediate01
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.utils.utils import normalize_angle, circular_mean
from maps.map_intermediate_01 import MyMapIntermediate01
from collections import deque
from drone_modules.occupancy_grid import OccupancyGrid
        
class ExplorationGrid():

    def __init__(self,drone,map_size=(400, 400), cell_size=40, **kwargs):

        self.counterStraight = 0
        self.angleStopTurning = 0
        self.isTurning = False
        self.iteration: int = 0
        self.drone = drone

        # Grille d'exploration
        self.cell_size = cell_size
        self.grid_size = (map_size[0] // cell_size, map_size[1] // cell_size)
        self.exploration_grid = np.zeros(self.grid_size, dtype=int)

        # Integrate the OccupancyGrid
        self.resolution = 8  # Adjust the resolution based on your needs
        self.grid = OccupancyGrid(size_area_world=drone.size_area, resolution=self.resolution, lidar=drone.lidar())

    def define_message_for_all(self):
        pass

    def control(self):
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}


        self.estimated_pose = Pose(np.asarray(self.drone.measured_gps_position()),
                                   self.drone.measured_compass_angle())
        
        # Update the occupancy grid with lidar data and current position
        self.grid.update_grid(self.estimated_pose)

        self.iteration += 1
        if self.iteration % 5 == 0:
            self.grid.display(self.grid.grid, self.estimated_pose, title="Occupancy Grid")
            self.grid.display(self.grid.zoomed_grid, self.estimated_pose, title="Zoomed Occupancy Grid")


        # Use the grid to find unexplored frontiers and guide exploration
        frontier_points = self.grid.find_frontier()

        if frontier_points:
            # Use the first frontier point to navigate the drone
            target_x, target_y = frontier_points[0]
            target_angle = math.atan2(target_y - self.drone.measured_gps_position()[1], 
                                      target_x - self.drone.measured_gps_position()[0])
            diff_angle = normalize_angle(target_angle - self.drone.measured_compass_angle())

            # Command to move towards the frontier
            kp = 2.0
            angular_speed = kp * diff_angle
            command["rotation"] = max(-1.0, min(1.0, angular_speed))

            if abs(angular_speed) > 0.5:
                command["forward"] = 0.2
            else:
                command["forward"] = 0.5

        command = self.control_exploration()

        return command

    def control_exploration(self):
        # Mettre à jour la grille d'exploration
        current_cell = self.get_current_cell()
        self.exploration_grid[current_cell] += 1

        # Trouver la cellule la moins visitée à proximité
        target_cell = self.find_least_visited_cell()

        # Calculer l'angle pour se diriger vers cette cellule
        target_angle = math.atan2(target_cell[1] - current_cell[1],
                                  target_cell[0] - current_cell[0])

        # Calculer l'angle nécessaire pour tourner
        diff_angle = normalize_angle(target_angle - self.drone.measured_compass_angle())

        # Commandes de mouvement
        command = {"forward": 0.5, "lateral": 0.0, "rotation": 0.0}
        kp = 2.0
        angular_speed = kp * diff_angle
        command["rotation"] = max(-1.0, min(1.0, angular_speed))

        # Réduire la vitesse si le drone doit beaucoup tourner
        if abs(angular_speed) > 0.5:
            command["forward"] = 0.2

        return command

    def get_current_cell(self):
        """
        Retourne la cellule actuelle basée sur la position du drone.
        """
        x, y = self.drone.measured_gps_position()
        cell_x = int((x + self.grid_size[0] * self.cell_size // 2) // self.cell_size)
        cell_y = int((y + self.grid_size[1] * self.cell_size // 2) // self.cell_size)
        return max(0, min(cell_x, self.grid_size[0] - 1)), max(0, min(cell_y, self.grid_size[1] - 1))

    def find_least_visited_cell(self, search_radius=2, memory_size=10):
        """
        Trouve la cellule la moins visitée dans un rayon de recherche.
        Évite que le drone ne revienne systématiquement sur ses pas en utilisant une mémoire des cellules visitées.

        :param search_radius: Rayon de recherche autour de la cellule actuelle.
        :param memory_size: Taille de la mémoire des cellules récemment visitées.
        """
        current_cell = self.get_current_cell()

        # Initialisation de la mémoire si elle n'existe pas encore
        if not hasattr(self, 'recently_visited_cells'):
            self.recently_visited_cells = deque(maxlen=memory_size)

        # Ajouter la cellule actuelle à la mémoire
        self.recently_visited_cells.append(current_cell)

        # Variables pour stocker la meilleure cellule
        min_score = float('inf')
        target_cell = current_cell

        # Parcourir toutes les cellules dans la zone de recherche
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                # Calculer les coordonnées de la cellule voisine
                nx, ny = current_cell[0] + dx, current_cell[1] + dy

                # Vérifier que la cellule est dans les limites de la grille
                if 0 <= nx < self.grid_size[0] and 0 <= ny < self.grid_size[1]:
                    # Ignorer les cellules récemment visitées
                    if (nx, ny) in self.recently_visited_cells:
                        continue

                    #Calculer un score basé sur le nombre de visites, la distance, et la pondération pour aller à gauche
                    visits = self.exploration_grid[nx, ny]
                    distance = math.sqrt(dx**2 + dy**2)  # Distance entre les cellules
                    left_bias = 5 * nx  # Pondération pour favoriser les cellules à gauche
                    score = visits - 0.25 * distance + left_bias  # Pondération combinée

                    # Mettre à jour la cellule cible si le score est meilleur
                    if score < min_score:
                        min_score = score
                        target_cell = (nx, ny)

        return target_cell

def main():
    
    my_map = MyMapIntermediate01()
    my_playground = my_map.construct_playground(drone_type=ExplorationGrid)

    # draw_semantic_rays : enable the visualization of the semantic rays
    gui = GuiSR(playground=my_playground,
                the_map=my_map,
                draw_semantic_rays=True,
                use_keyboard=False,
                )
    gui.run()


if __name__ == '__main__':
    main()