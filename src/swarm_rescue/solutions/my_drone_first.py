"""
First assessment controller
"""
import math
import arcade
from math import sqrt
from copy import deepcopy
from typing import Optional
from heapq import heappush, heappop
from random import randrange
from enum import Enum

import numpy as np

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.utils import normalize_angle, circular_mean
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle, sign
from spg_overlay.utils.timer import Timer
from spg_overlay.utils.path import Path
from spg_overlay.utils.pose import Pose
from spg_overlay.entities.keyboard_controller import KeyboardController

from drone_modules.slam_module import SLAMModule
from drone_modules.exploration_grid import ExplorationGrid
from drone_modules.path_planner import PathPlanner
from drone_modules.path_tracker import PathTracker
from drone_modules.occupancy_grid import OccupancyGrid

class MyDroneFirst(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        SEARCHING_WOUNDED = 1
        GRASPING_WOUNDED = 2
        SEARCHING_RESCUE_CENTER = 3
        DROPPING_AT_RESCUE_CENTER = 4
        FOLLOW_PATH = 5
        INITIAL = 99

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        
        self.state = self.Activity.INITIAL      #SEARCHING_WOUNDED

        # Modules
        self.slam = SLAMModule() #Ignorer pas encore implémenter
        self.grid = OccupancyGrid(size_area_world=self.size_area, resolution=8,drone=self)
        self.explorationGrid = ExplorationGrid(drone=self,grid=self.grid)
        #self.path_planner = PathPlanner(grid=self.grid,resolution=8) #Ignorer pas encore implémenter
        self.pathTracker = PathTracker()
        self.k = 0
        self.keyController = KeyboardController()

        self.pose = Pose()
        self.path = Path()
        self.timer = Timer(start_now=True)



    def control(self): # BOUCLE PRINCIPALE
        """
        Le drone utlise une Occupancy Grid pour explorer les zones non visitée 
        qui est inclus dans la classe ExplorationGrid qui s'occupe de l'exploration 
        de la grille. Lorsque la personne est trouvée nous utilisons process_semantic_sensor pour retourner à la base.

        """
        # Calcul du temps écoulé depuis la dernière itération (FPS)
        delta_time = self.timer.get_elapsed_time()
        self.timer.restart()  # Redémarrage pour le prochain cycle
        #self.explorationGrid.control()
        #print(self.grid.grid)

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper":0}

        self.pose.position = self.true_position()
        self.pose.orientation = self.true_angle()
        
        vx,vy = self.measured_velocity()
        v_angle = self.measured_angular_velocity()

        found_wounded, found_rescue_center, command_semantic = (self.process_semantic_sensor()) # command_semantic à ignorer
        
        # Transitions de la machine à états
        self.uptade_state(found_wounded,found_rescue_center)
        # Mise à jour & Affichage de l'Occupancy Grid
        self.grid.update_grid(self.pose)
        
        print(self.grid.grid)

        # Commandes selon l'état
        if self.state is self.Activity.SEARCHING_WOUNDED:
            command = self.explorationGrid.control()
            command["grasper"] = 0
        elif self.state is self.Activity.GRASPING_WOUNDED:
            command =  command_semantic # A Modifier par un path planner et tracker
            command["grasper"] = 1
        elif self.state is self.Activity.SEARCHING_RESCUE_CENTER:
            command = self.explorationGrid.control() 
            command["grasper"] = 1
        elif self.state is self.Activity.DROPPING_AT_RESCUE_CENTER:
            command = command_semantic # A Modifier par un path_planner et tracker
            command["grasper"] = 1
        
        # Etat intermédiaire pour les tests: Créer son chemin
        if self.state is self.Activity.INITIAL:
            command["grasper"] = 0
            pose0 = Pose(self.true_position(),self.true_angle())
            pose1 = Pose(np.array([295,50,-np.pi/2]))
            pose2 = Pose(np.array([230,50,-np.pi/2]))
            pose3 = Pose(np.array([230,-100,-np.pi/2]))
            self.path.append(pose0)
            self.path.append(pose1)
            self.path.append(pose2)
            self.path.append(pose3)
            self.state = self.Activity.FOLLOW_PATH
        elif self.state is self.Activity.FOLLOW_PATH:
            command = self.pathTracker.control(self.pose,self.path,delta_time)
            # Si on a finis le chemin
            if self.pathTracker.isFinish(self.path):
                print("finish")
                self.path.reset()
                self.state = self.Activity.FOLLOW_PATH # A changer
        return command
    
    def draw_bottom_layer(self):
        #self.draw_setpoint()
        self.draw_path(path=self.path, color=(255, 0, 255))
        self.draw_direction()
        #self.draw_antedirection()
        return None
    

    def draw_setpoint(self):
        half_width = self._half_size_array[0]
        half_height = self._half_size_array[1]
        pt1 = self.position_setpoint + np.array([half_width, 0])
        pt2 = self.position_setpoint + np.array([half_width, 2 * half_height])
        arcade.draw_line(float(pt2[0]),
                            float(pt2[1]),
                            float(pt1[0]),
                            float(pt1[1]),
                            color=arcade.color.GRAY)

    def draw_path(self, path, color: list[int, int, int]):
        length = path.length()
        # print(length)
        pt2 = None
        for ind_pt in range(length):
            pose = path.get(ind_pt)
            pt1 = pose.position + self._half_size_array
            # print(ind_pt, pt1, pt2)
            if ind_pt > 0:
                arcade.draw_line(float(pt2[0]),
                                 float(pt2[1]),
                                 float(pt1[0]),
                                 float(pt1[1]), color)
            pt2 = pt1
    
    def draw_direction(self):
        pt1 = np.array([self.true_position()[0], self.true_position()[1]])
        pt1 = pt1 + self._half_size_array
        pt2 = pt1 + 250 * np.array([math.cos(self.true_angle()),
                                    math.sin(self.true_angle())])
        color = (255, 64, 0)
        arcade.draw_line(float(pt2[0]),
                         float(pt2[1]),
                         float(pt1[0]),
                         float(pt1[1]),
                         color)

    def draw_antedirection(self):
        pt1 = np.array([self.true_position()[0], self.true_position()[1]])
        pt1 = pt1 + self._half_size_array
        pt2 = pt1 + 150 * np.array([math.cos(self.true_angle() + np.pi / 2),
                                    math.sin(self.true_angle() + np.pi / 2)])
        color = (255, 64, 0)
        arcade.draw_line(float(pt2[0]),
                         float(pt2[1]),
                         float(pt1[0]),
                         float(pt1[1]),
                         color)

    def process_lidar_sensor(self, the_lidar_sensor): 
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

    def define_message_for_all(self):#Ignorer
        """
        Define the message, the drone will send to and receive from other surrounding drones.
        """
        msg_data = (self.identifier,
                    (self.measured_gps_position(), self.measured_compass_angle()))
        return msg_data

    def process_communication_sensor(self): # Ignorer
        found_drone = False

        command_comm = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0}

        if self.communicator:
            received_messages = self.communicator.received_messages
            nearest_drone_coordinate1 = (
                self.measured_gps_position(), self.measured_compass_angle())
            nearest_drone_coordinate2 = deepcopy(nearest_drone_coordinate1)
            (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
            (nearest_position2, nearest_angle2) = nearest_drone_coordinate2

            min_dist1 = 10000
            min_dist2 = 10000
            diff_angle = 0

            # Search the two nearest drones around
            for msg in received_messages:
                message = msg[1]
                coordinate = message[1]
                (other_position, other_angle) = coordinate

                dx = other_position[0] - self.measured_gps_position()[0]
                dy = other_position[1] - self.measured_gps_position()[1]
                distance = math.sqrt(dx ** 2 + dy ** 2)

                # if another drone is near
                if distance < min_dist1:
                    min_dist2 = min_dist1
                    min_dist1 = distance
                    nearest_drone_coordinate2 = nearest_drone_coordinate1
                    nearest_drone_coordinate1 = coordinate
                    found_drone = True
                elif distance < min_dist2 and distance != min_dist1:
                    min_dist2 = distance
                    nearest_drone_coordinate2 = coordinate

            if not found_drone:
                return found_drone, command_comm

            # If we found at least 2 drones
            if found_drone and len(received_messages) >= 2:
                (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
                (nearest_position2, nearest_angle2) = nearest_drone_coordinate2
                diff_angle1 = normalize_angle(
                    nearest_angle1 - self.measured_compass_angle())
                diff_angle2 = normalize_angle(
                    nearest_angle2 - self.measured_compass_angle())
                # The mean of 2 angles can be seen as the angle of a vector, which
                # is the sum of the two unit vectors formed by the 2 angles.
                diff_angle = math.atan2(0.5 * math.sin(diff_angle1) + 0.5 * math.sin(diff_angle2),
                                        0.5 * math.cos(diff_angle1) + 0.5 * math.cos(diff_angle2))

            # If we found only 1 drone
            elif found_drone and len(received_messages) == 1:
                (nearest_position1, nearest_angle1) = nearest_drone_coordinate1
                diff_angle1 = normalize_angle(
                    nearest_angle1 - self.measured_compass_angle())
                diff_angle = diff_angle1

            # if you are far away, you get closer
            # heading < 0: at left
            # heading > 0: at right
            # base.angular_vel_controller : -1:left, 1:right
            # we are trying to align : diff_angle -> 0
            command_comm["rotation"] = sign(diff_angle)

            # Desired distance between drones
            desired_dist = 60

            d1x = nearest_position1[0] - self.measured_gps_position()[0]
            d1y = nearest_position1[1] - self.measured_gps_position()[1]
            distance1 = math.sqrt(d1x ** 2 + d1y ** 2)

            d1 = distance1 - desired_dist
            # We use a logistic function. -1 < intensity1(d1) < 1 and  intensity1(0) = 0
            # intensity1(d1) approaches 1 (resp. -1) as d1 approaches +inf (resp. -inf)
            intensity1 = 2 / (1 + math.exp(-d1 / (desired_dist * 0.5))) - 1

            direction1 = math.atan2(d1y, d1x)
            heading1 = normalize_angle(direction1 - self.measured_compass_angle())

            # The drone will slide in the direction of heading
            longi1 = intensity1 * math.cos(heading1)
            lat1 = intensity1 * math.sin(heading1)

            # If we found only 1 drone
            if found_drone and len(received_messages) == 1:
                command_comm["forward"] = longi1
                command_comm["lateral"] = lat1

            # If we found at least 2 drones
            elif found_drone and len(received_messages) >= 2:
                d2x = nearest_position2[0] - self.measured_gps_position()[0]
                d2y = nearest_position2[1] - self.measured_gps_position()[1]
                distance2 = math.sqrt(d2x ** 2 + d2y ** 2)

                d2 = distance2 - desired_dist
                intensity2 = 2 / (1 + math.exp(-d2 / (desired_dist * 0.5))) - 1

                direction2 = math.atan2(d2y, d2x)
                heading2 = normalize_angle(direction2 - self.measured_compass_angle())

                longi2 = intensity2 * math.cos(heading2)
                lat2 = intensity2 * math.sin(heading2)

                command_comm["forward"] = 0.5 * (longi1 + longi2)
                command_comm["lateral"] = 0.5 * (lat1 + lat2)

        return found_drone, command_comm
        
    def uptade_state(self,found_wounded,found_rescue_center):
                
        #############
        # TRANSITIONS OF THE STATE MACHINE
        #############

        if self.state is self.Activity.SEARCHING_WOUNDED and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        elif (self.state is self.Activity.GRASPING_WOUNDED and
              self.base.grasper.grasped_entities):
            self.state = self.Activity.SEARCHING_RESCUE_CENTER

        elif (self.state is self.Activity.GRASPING_WOUNDED and
              not found_wounded):
            self.state = self.Activity.SEARCHING_WOUNDED

        elif (self.state is self.Activity.SEARCHING_RESCUE_CENTER and
              found_rescue_center):
            self.state = self.Activity.DROPPING_AT_RESCUE_CENTER

        elif (self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and
              not self.base.grasper.grasped_entities):
            self.state = self.Activity.SEARCHING_WOUNDED

        elif (self.state is self.Activity.DROPPING_AT_RESCUE_CENTER and
              not found_rescue_center):
            self.state = self.Activity.SEARCHING_RESCUE_CENTER
        
        return None

    def update_command_search(self,command): #Ignorer
        command_lidar, collision_lidar = self.process_lidar_sensor(self.lidar())
        found, command_comm = self.process_communication_sensor()

        alpha = 0.4
        alpha_rot = 0.75
        if collision_lidar:
            alpha_rot = 0.1

        # The final command  is a combination of 2 commands
        command["forward"] = \
            alpha * command_comm["forward"] \
            + (1 - alpha) * command_lidar["forward"]
        command["lateral"] = \
            alpha * command_comm["lateral"] \
            + (1 - alpha) * command_lidar["lateral"]
        command["rotation"] = \
            alpha_rot * command_comm["rotation"] \
            + (1 - alpha_rot) * command_lidar["rotation"]
        
        return command

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move
        towards a wound person or the rescue center
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

        # Gére le déplacement aprés avoir trouvée la personne à ignorer
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