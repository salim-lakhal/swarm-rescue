"""
First assessment controller
"""
import math
from copy import deepcopy
from typing import Optional

import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, sign

class MyDroneFirst(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)

    def control(self): # BOUCLE PRINCIPALE
        """
        In this example, we only use the sensors sensor
        The idea is to make the drones move like a school of fish.
        The sensors will help avoid running into walls.
        """
        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0}

        command_lidar, collision_lidar = self.process_lidar_sensor(self.lidar())

        # if collision_lidar: # Gére la collision_lidar

        return command

    def process_lidar_sensor(self, the_lidar_sensor): #Gére les données du LIDAR à MODIFIER c'est un exemple
        command = {"forward": 1.0,
                   "lateral": 0.0,
                   "rotation": 0.0}
        angular_vel_controller = 1.0

        values = the_lidar_sensor.get_sensor_values()

        if values is None:
            return command, False

        ray_angles = the_lidar_sensor.ray_angles
        size = the_lidar_sensor.resolution

        far_angle_raw = 0
        near_angle_raw = 0
        min_dist = 1000
        if size != 0:
            # far_angle_raw : angle with the longer distance
            far_angle_raw = ray_angles[np.argmax(values)]
            min_dist = min(values)
            # near_angle_raw : angle with the nearest distance
            near_angle_raw = ray_angles[np.argmin(values)]

        far_angle = far_angle_raw
        # If far_angle_raw is small then far_angle = 0
        if abs(far_angle) < 1 / 180 * np.pi:
            far_angle = 0.0

        near_angle = near_angle_raw
        far_angle = normalize_angle(far_angle)

        # The drone will turn toward the zone with the more space ahead
        if size != 0:
            if far_angle > 0:
                command["rotation"] = angular_vel_controller
            elif far_angle == 0:
                command["rotation"] = 0
            else:
                command["rotation"] = -angular_vel_controller

        # If near a wall then 'collision' is True and the drone tries to turn its back to the wall
        collision = False
        if size != 0 and min_dist < 50:
            collision = True
            if near_angle > 0:
                command["rotation"] = -angular_vel_controller
            else:
                command["rotation"] = angular_vel_controller

        return command, collision

    #TODO : Gére la localisation et la cartographie de la carte
    def slam_update(self, sensor_data):
        # Implémentez votre algorithme SLAM ici
        pass

    # TODO : Exploration aléatoire lié à process_lidar_sensor
    def exploration_logic(self, current_position, map_data):
        # Implémentez Wall-Following ou une autre logique
        pass

    #TODO : Créer la fonction qui planifie le chemin sachant une collection de point accesible sur la carte
    def plan_path(self, start, goal):
        # Implémenter A* ici
        pass

    # TODO : Comment on va suivre le chemin avec le drone
    def follow_path(self, path):
        # Implémenter Pure Pursuit ici
        pass