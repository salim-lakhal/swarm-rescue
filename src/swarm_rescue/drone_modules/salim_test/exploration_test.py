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


    
class OccupancyGrid(Grid):
    """Simple occupancy grid"""

    def __init__(self,
                 size_area_world,
                 resolution: float,
                 lidar):
        super().__init__(size_area_world=size_area_world,
                         resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.lidar = lidar

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution
                                   + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution
                                   + 0.5)

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.zoomed_grid = np.empty((self.x_max_grid, self.y_max_grid))

    def find_frontier(self):
        """
        Find frontier cells in the occupancy grid.
        A frontier is a free cell adjacent to an unknown cell.
        """
        frontier_points = []

        # Loop through each cell in the grid
        for x in range(1, self.grid.shape[0] - 1):  # Exclude borders to avoid out-of-bounds
            for y in range(1, self.grid.shape[1] - 1):
                if self.grid[x, y] == -4.0:  # Free cell
                    # Check if any adjacent cell is unknown (0)
                    neighbors = self.grid[x-1:x+2, y-1:y+2]
                    if np.any(neighbors == 0):  # There is an unknown cell around
                        frontier_points.append((x, y))

        # Convert grid indices to world coordinates
        return [
            (pt[0] * self.resolution, pt[1] * self.resolution)
            for pt in frontier_points
        ]
    
    def update_grid(self, pose: Pose):
        """
        Bayesian map update with new observation
        lidar : lidar data
        pose : corrected pose in world coordinates
        """
        EVERY_N = 3
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.602
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = -4.0
        THRESHOLD_MIN = -40
        THRESHOLD_MAX = 40

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        # Compute cos and sin of the absolute angle of the lidar
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # For empty zones
        # points_x and point_y contains the border of detected empty zone
        # We use a value a little bit less than LIDAR_DIST_CLIP because of the
        # noise in lidar
        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        # All values of lidar_dist_empty_clip are now <= max_range
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip,
                                                  cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip,
                                                  sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1],
                                      pt_x, pt_y,
                                      EMPTY_ZONE_VALUE)

        # For obstacle zones, all values of lidar_dist are < max_range
        select_collision = lidar_dist < max_range

        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

        # the current position of the drone is free !
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # threshold values
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

        # compute zoomed grid for displaying
        self.zoomed_grid = self.grid.copy()
        new_zoomed_size = (int(self.size_area_world[1] * 0.5),
                           int(self.size_area_world[0] * 0.5))
        self.zoomed_grid = cv2.resize(self.zoomed_grid, new_zoomed_size,
                                      interpolation=cv2.INTER_NEAREST)
        
class MyDroneSemantic(DroneAbstract):
    class Activity(Enum):
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4

    def __init__(self, identifier: Optional[int] = None, map_size=(400, 400), cell_size=40, **kwargs):
        super().__init__(identifier=identifier, display_lidar_graph=False, **kwargs)
        self.state = self.Activity.SEARCHING_WOUNDED
        self.counterStraight = 0
        self.angleStopTurning = 0
        self.isTurning = False
        self.iteration: int = 0

        # Grille d'exploration
        self.cell_size = cell_size
        self.grid_size = (map_size[0] // cell_size, map_size[1] // cell_size)
        self.exploration_grid = np.zeros(self.grid_size, dtype=int)

        # Integrate the OccupancyGrid
        self.resolution = 8  # Adjust the resolution based on your needs
        self.grid = OccupancyGrid(size_area_world=self.size_area, resolution=self.resolution, lidar=self.lidar())

    def define_message_for_all(self):
        pass

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move
        towards a wounded person or the rescue center
        """
        command = {"forward": 0.5,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller_max = 1.0

        detection_semantic = self.semantic_values()
        best_angle = 0

        found_wounded = False
        if (self.state is self.Activity.SEARCHING_WOUNDED
            or self.state is self.Activity.GRASPING_WOUNDED) \
                and detection_semantic is not None:
            scores = []
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if (data.entity_type ==
                        DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and
                        not data.grasped):
                    found_wounded = True
                    v = (data.angle * data.angle) + \
                        (data.distance * data.distance / 10 ** 5)
                    scores.append((v, data.angle, data.distance))

            # Select the best one among wounded persons detected
            best_score = 10000
            for score in scores:
                if score[0] < best_score:
                    best_score = score[0]
                    best_angle = score[1]

        found_rescue_center = False
        is_near = False
        angles_list = []
        if (self.state is self.Activity.SEARCHING_RESCUE_CENTER
            or self.state is self.Activity.DROPPING_AT_RESCUE_CENTER) \
                and detection_semantic:
            for data in detection_semantic:
                if (data.entity_type ==
                        DroneSemanticSensor.TypeEntity.RESCUE_CENTER):
                    found_rescue_center = True
                    angles_list.append(data.angle)
                    is_near = (data.distance < 30)

            if found_rescue_center:
                best_angle = circular_mean(np.array(angles_list))

        if found_rescue_center or found_wounded:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

        if found_rescue_center and is_near:
            command["forward"] = 0.0
            command["rotation"] = -1.0

        return found_wounded, found_rescue_center, command

    def control(self):
        command = {"forward": 0.0, "lateral": 0.0, "rotation": 0.0, "grasper": 0}


        self.estimated_pose = Pose(np.asarray(self.measured_gps_position()),
                                   self.measured_compass_angle())
        
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
            target_angle = math.atan2(target_y - self.measured_gps_position()[1], 
                                      target_x - self.measured_gps_position()[0])
            diff_angle = normalize_angle(target_angle - self.measured_compass_angle())

            # Command to move towards the frontier
            kp = 2.0
            angular_speed = kp * diff_angle
            command["rotation"] = max(-1.0, min(1.0, angular_speed))

            if abs(angular_speed) > 0.5:
                command["forward"] = 0.2
            else:
                command["forward"] = 0.5

        
        found_wounded, found_rescue_center, command_semantic = self.process_semantic_sensor()

        # Transitions de la machine à états
        if self.state is self.Activity.SEARCHING_WOUNDED and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED
        elif self.state is self.Activity.GRASPING_WOUNDED and self.base.grasper.grasped_entities:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER
        elif self.state is self.Activity.GRASPING_WOUNDED and not found_wounded:
            self.state = self.Activity.SEARCHING_WOUNDED
        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER and found_rescue_center:
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER
        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not self.base.grasper.grasped_entities:
            self.state = self.Activity.SEARCHING_WOUNDED
        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and not found_rescue_center:
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        # Commandes selon l'état
        if self.state is self.Activity.SEARCHING_WOUNDED:
            command = self.control_exploration()
            command["grasper"] = 0
        elif self.state is self.Activity.GRASPING_WOUNDED:
            command = command_semantic
            command["grasper"] = 1
        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            command = self.control_exploration()
            command["grasper"] = 1
        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic
            command["grasper"] = 1
        
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
        diff_angle = normalize_angle(target_angle - self.measured_compass_angle())

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
        x, y = self.measured_gps_position()
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
    my_playground = my_map.construct_playground(drone_type=MyDroneSemantic)

    # draw_semantic_rays : enable the visualization of the semantic rays
    gui = GuiSR(playground=my_playground,
                the_map=my_map,
                draw_semantic_rays=True,
                use_keyboard=False,
                )
    gui.run()


if __name__ == '__main__':
    main()